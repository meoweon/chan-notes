"""抓取币安 U 本位永续 30m 历史 K 线（data.binance.vision 月度打包，走本地代理）。

用法：
    python scripts/fetch_binance.py [--start 2019-09] [--end 2026-10]

合规与稳健：
- 单线程顺序下载，请求间隔 0.15s；
- 断点续跑：已存在 parquet 的币种跳过，月粒度失败重试 2 次；
- 2025+ 的 zip 有表头行、open_time 可能为微秒，均自动处理；
- 时间戳为 UTC（naive），加密 24/7 连续交易。
"""
import argparse
import io
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 数据根目录在脚本内硬编码设置（避免 shell 传含中文的环境变量时编码损毁导致文件蒸发的坑）
BARS_ROOT = str(Path(__file__).resolve().parents[1] / "data" / "bars_crypto")
os.environ["CHAN_NOTES_BARS"] = BARS_ROOT

import pandas as pd

from chan_notes.data import store

PROXY = "http://127.0.0.1:7890"
BASE = "https://data.binance.vision/data/futures/um/monthly/klines/{sym}/{freq}/{sym}-{freq}-{ym}.zip"

# 高流动性、长历史的 U 本位永续（上市月均 >= 2020-09）
UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "LTCUSDT", "BCHUSDT", "EOSUSDT", "TRXUSDT", "ETCUSDT",
    "LINKUSDT", "XLMUSDT", "DOGEUSDT", "SOLUSDT", "ATOMUSDT",
]

COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]

_opener = urllib.request.build_opener(
    urllib.request.ProxyHandler({"http": PROXY, "https": PROXY}))
_opener.addheaders = [("User-Agent", "Mozilla/5.0")]


def fetch_month(sym: str, ym: str, freq: str) -> pd.DataFrame | None:
    """返回该月 K 线；404 返回 None；其他错误抛异常。"""
    url = BASE.format(sym=sym, ym=ym, freq=freq)
    try:
        raw = _opener.open(url, timeout=30).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    zf = zipfile.ZipFile(io.BytesIO(raw))
    name = zf.namelist()[0]
    text = zf.read(name).decode("utf-8")
    # 新版文件带表头行
    first = text.lstrip().split("\n", 1)[0]
    if first.startswith("open_time"):
        text = text.split("\n", 1)[1]
    df = pd.read_csv(io.StringIO(text), header=None, names=COLS)
    if df.empty:
        return None
    ts = df["open_time"].astype("int64")
    unit = "us" if ts.iloc[0] > 10**14 else "ms"
    df["time"] = pd.to_datetime(ts, unit=unit, utc=True).dt.tz_localize(None)
    return df[["time", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float})


def months_between(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-09")
    ap.add_argument("--end", default="2026-10")
    ap.add_argument("--sleep", type=float, default=0.15)
    ap.add_argument("--freq", default="30m")
    ap.add_argument("--symbols", default="", help="comma separated, default all")
    args = ap.parse_args()
    freq = args.freq

    universe = args.symbols.split(",") if args.symbols else UNIVERSE
    yms = [f"{p.year}-{p.month:02d}" for p in pd.period_range(args.start, args.end, freq="M")]
    print(f"universe={len(universe)} months={len(yms)} freq={freq}", flush=True)

    ok = fail = 0
    for si, sym in enumerate(universe, 1):
        if store.has(sym, freq):
            print(f"[{si}/{len(universe)}] {sym} exists, skip", flush=True)
            continue
        parts = []
        for ym in yms:
            for attempt in (0, 1, 2):
                try:
                    df = fetch_month(sym, ym, freq)
                    break
                except Exception as e:
                    print(f"  {sym} {ym} attempt{attempt} ERR {type(e).__name__}: {e}", flush=True)
                    time.sleep(5 * (attempt + 1))
                    df = "ERR"
            if isinstance(df, str):
                fail += 1
                continue
            if df is not None:
                parts.append(df)
            time.sleep(args.sleep)
        if parts:
            all_df = pd.concat(parts, ignore_index=True)
            store.save(sym, all_df, freq)
            ok += 1
            print(f"[{si}/{len(UNIVERSE)}] {sym} bars={len(all_df)} "
                  f"{all_df.time.iloc[0]} ~ {all_df.time.iloc[-1]}", flush=True)
        else:
            print(f"[{si}/{len(UNIVERSE)}] {sym} NO DATA", flush=True)
    print(f"DONE ok={ok} month_fail={fail}", flush=True)


if __name__ == "__main__":
    main()
