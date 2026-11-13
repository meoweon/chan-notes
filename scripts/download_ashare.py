"""从 baostock 批量下载 A 股 K 线并保存为 chan-notes parquet。

baostock 规则：禁止多连接，单会话限速。
用法:
    python scripts/download_ashare.py
    python scripts/download_ashare.py --start 2023-01-01 --end 2026-11-09
"""
import argparse
import sys
import time
from pathlib import Path

import baostock as bs
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chan_notes.data import store

DEFAULT_STOCKS = [
    "sh.600000",  # 浦发银行
    "sh.600036",  # 招商银行
    "sh.601318",  # 中国平安
    "sz.000001",  # 平安银行
    "sz.000858",  # 五粮液
    "sh.600519",  # 贵州茅台
    "sh.601398",  # 工商银行
    "sh.601939",  # 建设银行
    "sh.600276",  # 恒瑞医药
    "sh.600030",  # 中信证券
    "sz.000651",  # 格力电器
    "sh.601888",  # 中国中免
    "sz.300750",  # 宁德时代
    "sh.600900",  # 长江电力
    "sh.601012",  # 隆基绿能
    "sz.002594",  # 比亚迪
    "sh.600809",  # 山西汾酒
    "sz.000333",  # 美的集团
    "sh.601899",  # 紫金矿业
    "sz.002475",  # 立讯精密
]

FREQ_MAP = {"15m": "15", "30m": "30", "1h": "60"}


def _drain(rs) -> list[list[str]]:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    return rows


def fetch_intraday(code: str, freq: str, start: str, end: str) -> pd.DataFrame:
    """抓取单只股票的某个周期数据。"""
    fields = "date,time,code,open,high,low,close,volume,amount"
    rs = bs.query_history_k_data_plus(
        code, fields,
        start_date=start, end_date=end,
        frequency=freq, adjustflag="2",
    )
    rows = _drain(rs)
    if not rows:
        return pd.DataFrame(columns=["code", "time", "open", "high", "low", "close", "volume", "amount"])

    df = pd.DataFrame(rows, columns=fields.split(","))
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S%f")
    df = df.drop(columns=["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _quarter_ranges(start: str, end: str) -> list[tuple[str, str]]:
    """将日期范围拆成季度。"""
    import calendar
    from datetime import datetime, timedelta
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    quarters = []
    cur = s
    while cur < e:
        q_end_month = ((cur.month - 1) // 3 + 1) * 3
        q_end = cur.replace(month=q_end_month, day=1)
        last_day = calendar.monthrange(q_end.year, q_end.month)[1]
        q_end = q_end.replace(day=last_day)
        actual_end = min(q_end, e)
        quarters.append((cur.strftime("%Y-%m-%d"), actual_end.strftime("%Y-%m-%d")))
        cur = q_end + timedelta(days=1)
    return quarters


def fetch_intraday_chunked(code: str, freq: str, start: str, end: str, delay: float = 0.3) -> pd.DataFrame:
    """按季度分段抓取，避免单次查询过大。"""
    chunks = _quarter_ranges(start, end)
    print(f"    共 {len(chunks)} 个季度分段", flush=True)
    frames = []
    for i, (qs, qe) in enumerate(chunks):
        print(f"    Q{i+1}: {qs} ~ {qe}...", end="", flush=True)
        df = fetch_intraday(code, freq, qs, qe)
        if not df.empty:
            frames.append(df)
            print(f" {len(df)} bars", flush=True)
        else:
            print(" 空", flush=True)
        if i < len(chunks) - 1:
            time.sleep(delay)
    if not frames:
        return pd.DataFrame(columns=["code", "time", "open", "high", "low", "close", "volume", "amount"])
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    return result


def convert_to_parquet(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """转为 chan-notes parquet 格式。"""
    return pd.DataFrame({
        "time": df["time"],
        "code": code,
        "open": df["open"],
        "high": df["high"],
        "low": df["low"],
        "close": df["close"],
        "volume": df["volume"],
        "amount": df["amount"],
        "adjustflag": "2",
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks", default=",".join(DEFAULT_STOCKS))
    ap.add_argument("--timeframes", default="15m,30m,1h")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-11-09")
    ap.add_argument("--delay", type=float, default=0.3, help="请求间隔秒数")
    args = ap.parse_args()

    stocks = [s.strip() for s in args.stocks.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]

    print(f"计划下载: {len(stocks)} 只 A 股 × {len(timeframes)} 周期")
    print(f"股票: {stocks}")
    print(f"范围: {args.start} ~ {args.end}")
    print()

    lg = bs.login()
    if lg.error_code != "0":
        raise SystemExit(f"baostock 登录失败: {lg.error_msg}")

    t0 = time.time()
    total_bars = 0
    results = []

    try:
        for tf in timeframes:
            bs_freq = FREQ_MAP[tf]
            print(f"=== {tf} (baostock freq={bs_freq}) ===")
            for code in stocks:
                print(f"  {code}...", end=" ", flush=True)
                df = fetch_intraday_chunked(code, bs_freq, args.start, args.end, delay=args.delay)
                if df.empty:
                    print("无数据")
                    time.sleep(args.delay)
                    continue
                out = convert_to_parquet(df, code)
                store.save(code, out, tf)
                n = len(out)
                total_bars += n
                print(f"{n} bars [{out['time'].min()} ~ {out['time'].max()}]")
                results.append((code, tf, n))
                time.sleep(args.delay)
    finally:
        bs.logout()

    elapsed = time.time() - t0
    print(f"\n完成: {len(results)} 个组合, 共 {total_bars} bars, 耗时 {elapsed:.1f}s")

    # 保存元数据
    meta_path = Path(__file__).resolve().parents[1] / "data" / "meta"
    meta_path.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "baostock",
        "download_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": stocks,
        "timeframes": timeframes,
        "range": [args.start, args.end],
        "total_bars": total_bars,
        "elapsed_seconds": round(elapsed, 1),
        "details": [{"code": c, "timeframe": t, "bars": n} for c, t, n in results],
    }
    import json
    with open(meta_path / "baostock_download.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"元数据: {meta_path / 'baostock_download.json'}")


if __name__ == "__main__":
    main()
