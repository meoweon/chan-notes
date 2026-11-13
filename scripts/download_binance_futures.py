"""从 Binance 数据归档批量下载合约 K 线并导入 chan-notes parquet。

用法:
    python scripts/download_binance_futures.py \
        --symbols ALGOUSDT,AAVEUSDT \
        --timeframes 15m,1h,4h \
        --start 2020-01 --end 2026-10
"""
import argparse
import io
import ssl
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chan_notes.data import store

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def _month_range(start: str, end: str) -> list[str]:
    """生成 YYYY-MM 列表。"""
    s = datetime.strptime(start, "%Y-%m")
    e = datetime.strptime(end, "%Y-%m")
    months = []
    cur = s
    while cur <= e:
        months.append(cur.strftime("%Y-%m"))
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return months


def _download_month(symbol: str, interval: str, month: str) -> pd.DataFrame | None:
    """下载单月数据，返回 DataFrame 或 None（404/空）。"""
    url = f"{BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=30, context=_ctx)
        data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None

    if not data:
        return None

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = zf.namelist()
        if not names:
            return None
        frames = []
        for name in names:
            csv_data = zf.read(name)
            if not csv_data.strip():
                continue
            df = pd.read_csv(io.BytesIO(csv_data), header=None)
            if df.iloc[0, 0] == "open_time":
                df = df.iloc[1:]
            frames.append(df)
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)


def download_symbol(symbol: str, interval: str, months: list[str], max_workers: int = 8) -> pd.DataFrame:
    """并发下载一个交易对的全部月份。"""
    all_frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_download_month, symbol, interval, m): m for m in months}
        done_count = 0
        for fut in as_completed(futures):
            done_count += 1
            month = futures[fut]
            try:
                df = fut.result()
                if df is not None and len(df) > 0:
                    all_frames.append((month, df))
            except Exception as e:
                print(f"  WARN {symbol} {interval} {month}: {e}")
            if done_count % 12 == 0 or done_count == len(months):
                print(f"  进度: {done_count}/{len(months)} 月")
    if not all_frames:
        return pd.DataFrame()
    all_frames.sort(key=lambda x: x[0])
    raw = pd.concat([f for _, f in all_frames], ignore_index=True)
    raw[0] = pd.to_numeric(raw[0], errors="coerce")
    raw = raw.dropna(subset=[0])
    raw = raw.sort_values(0).drop_duplicates(subset=[0]).reset_index(drop=True)
    return raw


def convert_to_parquet(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """转为 chan-notes parquet 格式。"""
    out = pd.DataFrame({
        "time": pd.to_datetime(raw[0], unit="ms", utc=True).dt.tz_localize(None),
        "code": symbol,
        "open": raw[1].astype("float64"),
        "high": raw[2].astype("float64"),
        "low": raw[3].astype("float64"),
        "close": raw[4].astype("float64"),
        "volume": raw[5].astype("float64"),
        "amount": raw[7].astype("float64") if raw.shape[1] > 7 else 0.0,
        "adjustflag": "2",
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="ALGOUSDT,AAVEUSDT,CRVUSDT,MATICUSDT,SANDUSDT,MANAUSDT,AXSUSDT,GALAUSDT,ENJUSDT,COMPUSDT")
    ap.add_argument("--timeframes", default="15m,1h,4h")
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default="2026-10")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="只检查不下载")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]
    months = _month_range(args.start, args.end)

    print(f"计划下载: {len(symbols)} 币 × {len(timeframes)} 周期 × {len(months)} 月 = {len(symbols) * len(timeframes) * len(months)} 请求")
    print(f"币种: {symbols}")
    print(f"周期: {timeframes}")
    print(f"范围: {args.start} ~ {args.end}")
    print()

    if args.dry_run:
        return

    t0 = time.time()
    total_bars = 0
    results = []

    for tf in timeframes:
        print(f"=== {tf} ===")
        for sym in symbols:
            print(f"  {sym}...", end=" ", flush=True)
            raw = download_symbol(sym, tf, months, max_workers=args.workers)
            if raw.empty:
                print("无数据")
                continue
            out = convert_to_parquet(raw, sym)
            store.save(sym, out, tf)
            n = len(out)
            total_bars += n
            print(f"{n} bars [{out['time'].min()} ~ {out['time'].max()}]")
            results.append((sym, tf, n))

    elapsed = time.time() - t0
    print(f"\n完成: {len(results)} 个组合, 共 {total_bars} bars, 耗时 {elapsed:.1f}s")

    # 保存元数据
    meta_path = Path(__file__).resolve().parents[1] / "data" / "meta"
    meta_path.mkdir(parents=True, exist_ok=True)
    meta = {
        "source": "binance_futures",
        "download_time": datetime.now().isoformat(),
        "symbols": symbols,
        "timeframes": timeframes,
        "range": [args.start, args.end],
        "total_bars": total_bars,
        "elapsed_seconds": round(elapsed, 1),
        "details": [{"symbol": s, "timeframe": t, "bars": n} for s, t, n in results],
    }
    import json
    with open(meta_path / "binance_download.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"元数据: {meta_path / 'binance_download.json'}")


if __name__ == "__main__":
    main()
