"""把 Binance 期货 daily 归档增量补进已有 bars（只追加，绝不覆盖历史）。

用途：月末尚未过完时 monthly 归档还是 404，只有 daily 可得，用它补最近几天。

与 download_binance_futures.py 的区别：后者按月份整表重写，`store.save` 是
整文件覆盖——拿它单独下 11 月会把 2020-01~2026-10 全冲掉。本脚本读旧表、
拼接、按 time 去重后再原子替换。

用法:
    python scripts/backfill_binance_daily.py --start-date 2026-11-01 --end-date 2026-11-08
    python scripts/backfill_binance_daily.py --symbols BTCUSDT --dry-run
"""
import argparse
import io
import json
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chan_notes.data import store  # noqa: E402

BASE_URL = "https://data.binance.vision/data/futures/um/daily/klines"
COLS = ["time", "code", "open", "high", "low", "close", "volume", "amount", "adjustflag"]

_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def _dates(start: str, end: str) -> list[str]:
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    while d0 <= d1:
        out.append(d0.isoformat())
        d0 += timedelta(days=1)
    return out


def _fetch_day(symbol: str, interval: str, day: str) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{day}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        payload = urllib.request.urlopen(req, timeout=30, context=_ctx).read()
    except urllib.error.HTTPError as e:
        return None if e.code == 404 else _raise(e)
    except Exception as e:
        return _raise(e)
    if not payload:
        return None
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        frames = []
        for name in zf.namelist():
            raw = zf.read(name)
            if not raw.strip():
                continue
            df = pd.read_csv(io.BytesIO(raw), header=None)
            if df.iloc[0, 0] == "open_time":
                df = df.iloc[1:]
            frames.append(df)
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)
    raw[0] = pd.to_numeric(raw[0], errors="coerce")
    raw = raw.dropna(subset=[0])
    return raw.sort_values(0).drop_duplicates(subset=[0]).reset_index(drop=True)


def _raise(e):
    raise e


def _convert(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.shape[1] <= 7:
        raise ValueError(f"{symbol}: 归档只有 {raw.shape[1]} 列，缺成交额（第 7 列），拒绝写入假 amount")
    return pd.DataFrame({
        "time": pd.to_datetime(raw[0], unit="ms", utc=True).dt.tz_localize(None),
        "code": symbol,
        "open": raw[1].astype("float64"),
        "high": raw[2].astype("float64"),
        "low": raw[3].astype("float64"),
        "close": raw[4].astype("float64"),
        "volume": raw[5].astype("float64"),
        "amount": raw[7].astype("float64"),
        "adjustflag": "2",
    })


def _atomic_merge(symbol: str, tf: str, new: pd.DataFrame) -> tuple[int, int, int]:
    """读旧表 → 拼接去重 → 校验 → 原子替换。返回 (旧行数, 新行数, 与旧表重叠行数)。"""
    p = store._path(symbol, tf)
    if not p.exists():
        raise FileNotFoundError(f"{p} 不存在：本脚本只追加，不新建标的")
    old = pd.read_parquet(p)
    before = len(old)
    overlap = int(new["time"].isin(old["time"]).sum())

    merged = pd.concat([old, new], ignore_index=True)
    merged = merged.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    assert list(merged.columns) == COLS, f"{symbol} {tf}: 列序不符 {list(merged.columns)}"
    assert merged["time"].is_monotonic_increasing, f"{symbol} {tf}: 时间非单调"
    assert len(merged) >= before, f"{symbol} {tf}: 合并后行数反而变少（{before} → {len(merged)}）"
    assert set(old["time"]) <= set(merged["time"]), f"{symbol} {tf}: 旧 bar 的 time 有丢失"
    ohlc = merged[["open", "high", "low", "close"]]
    assert int(ohlc.isna().sum().sum()) == 0, f"{symbol} {tf}: OHLC 出现 NaN"
    bad = merged[(merged["volume"] > 0) & (merged["amount"] <= 0)]
    assert len(bad) == 0, f"{symbol} {tf}: {len(bad)} 行有量无额"

    tmp = p.with_suffix(".parquet.tmp")
    merged.to_parquet(tmp, index=False, engine="pyarrow")
    chk = pd.read_parquet(tmp)
    if len(chk) != len(merged):
        raise AssertionError(f"{symbol} {tf}: 回读行数不符，放弃替换")
    import os
    os.replace(tmp, p)
    return before, len(merged), overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default="2026-11-01")
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--timeframes", default="15m,1h,4h")
    ap.add_argument("--symbols", default="", help="留空=取 bars 目录下已有的全部 USDT 标的")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.symbols.strip():
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = sorted(f.stem for f in (ROOT / "data" / "bars" / "1h").glob("*USDT.parquet"))
    tfs = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    days = _dates(args.start_date, args.end_date)

    print(f"计划: {len(symbols)} 币 × {len(tfs)} 周期 × {len(days)} 日 = {len(symbols)*len(tfs)*len(days)} 请求")
    print(f"日期: {days[0]} ~ {days[-1]}")
    if args.dry_run:
        print("(dry-run，未联网、未写盘)")
        return

    t0 = time.time()
    details, missing, added_total = [], [], 0
    for tf in tfs:
        print(f"=== {tf} ===")
        for sym in symbols:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                got = list(pool.map(lambda d: (d, _fetch_day(sym, tf, d)), days))
            frames, absent = [], [d for d, f in got if f is None]
            for _, f in got:
                if f is not None and len(f):
                    frames.append(f)
            if not frames:
                missing.extend(f"{sym} {tf} {d}" for d in absent)
                print(f"  {sym:<14} 无可用归档（{len(absent)} 天 404）")
                continue
            new = _convert(pd.concat(frames, ignore_index=True), sym)
            before, after, overlap = _atomic_merge(sym, tf, new)
            delta = after - before
            added_total += delta
            last = pd.read_parquet(store._path(sym, tf), columns=["time"])["time"].max()
            details.append({"symbol": sym, "timeframe": tf, "bars_before": before,
                            "bars_after": after, "bars_added": delta, "bars_overlap": overlap,
                            "new_max_time": str(last)})
            print(f"  {sym:<14} +{delta:<6} → {after} bars，末条 {last}" + (f"（覆盖同刻 {overlap} 行）" if overlap else ""))
            if absent:
                missing.extend(f"{sym} {tf} {d}" for d in absent)
                print(f"                 缺日: {','.join(absent)}")

    elapsed = round(time.time() - t0, 1)
    meta = {
        "source": "binance_futures_daily",
        "download_time": datetime.now().isoformat(),
        "range": [args.start_date, args.end_date],
        "timeframes": tfs,
        "symbols": symbols,
        "bars_added_total": added_total,
        "elapsed_seconds": elapsed,
        "missing_days": missing,
        "details": details,
    }
    out = ROOT / "data" / "meta" / "binance_daily_backfill.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n完成: {len(details)} 个组合，新增 {added_total} bars，耗时 {elapsed}s")
    if missing:
        print(f"缺档 {len(missing)} 项: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    print(f"元数据: {out}")


if __name__ == "__main__":
    main()
