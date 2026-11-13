"""从 kline-data-downloader 的 CSV 导入到 chan-notes 的 parquet。

kline-data-downloader 数据格式：
  - 按天拆分：kline-data/{category}/{SYMBOL}/{timeframe}/{yyyy-MM-dd}.csv
  - 无表头：timestamp_ms,open,high,low,close,volume[,amount]
    第 7 列（quote asset volume，成交额）可选；缺列时本脚本拒绝导入，
    不再补 0.0——静默补 0 曾把 21 币的 amount 全部写成 0（见 D034/D035）。

chan-notes 数据格式：
  - 单股单文件：data/bars/{timeframe}/{SYMBOL}.parquet
  - 列：time(datetime64), code(str), open, high, low, close, volume, amount, adjustflag(str)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chan_notes.data import store

DEFAULT_KLINE_ROOT = Path(__file__).resolve().parents[2] / "kline-data-dowloader-cs" / "kline-data"

BASE_COLS = ["timestamp_ms", "open", "high", "low", "close", "volume"]


def _read_one(path: Path) -> pd.DataFrame:
    """读单个无表头 CSV；存在第 7 列时按成交额（quote asset volume）命名。"""
    df = pd.read_csv(path, header=None)
    if df.shape[1] < len(BASE_COLS):
        raise ValueError(f"{path}: 只有 {df.shape[1]} 列，缺少必需量价列")
    names = BASE_COLS + (["amount"] if df.shape[1] > len(BASE_COLS) else [])
    return df.set_axis(names, axis=1)


def import_symbol(category: str, symbol: str, timeframe: str, kline_root: Path, verbose: bool = True) -> int:
    """导入单个交易对的某个周期。

    Returns:
        导入的 K 线总数
    """
    src_dir = kline_root / category / symbol / timeframe
    if not src_dir.exists():
        raise FileNotFoundError(f"目录不存在: {src_dir}")

    csv_files = sorted(src_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未找到 CSV 文件: {src_dir}")

    frames = [_read_one(f) for f in csv_files]
    missing = [f for f, fr in zip(csv_files, frames) if "amount" not in fr.columns]
    if missing:
        raise ValueError(
            f"{symbol} {timeframe}: 源 CSV 无成交额列（第 7 列），拒绝写入假 amount；"
            f"示例 {missing[0]}（共 {len(missing)} 个文件）。"
            f"需要真成交额请改用 scripts/download_binance_futures.py"
        )

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.sort_values("timestamp_ms").drop_duplicates(subset=["timestamp_ms"]).reset_index(drop=True)

    out = pd.DataFrame({
        "time": pd.to_datetime(raw["timestamp_ms"], unit="ms", utc=True).dt.tz_localize(None),
        "code": symbol,
        "open": raw["open"].astype("float64"),
        "high": raw["high"].astype("float64"),
        "low": raw["low"].astype("float64"),
        "close": raw["close"].astype("float64"),
        "volume": raw["volume"].astype("float64"),
        "amount": raw["amount"].astype("float64"),
        "adjustflag": "2",
    })

    store.save(symbol, out, timeframe)
    if verbose:
        print(f"  {symbol} {timeframe}: {len(out)} bars  [{out['time'].min()} ~ {out['time'].max()}]")
    return len(out)


def main():
    ap = argparse.ArgumentParser(description="导入 kline-data-downloader CSV 到 chan-notes parquet")
    ap.add_argument("--category", default="future", choices=["spot", "future"], help="数据类型")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT", help="交易对，逗号分隔")
    ap.add_argument("--timeframes", default="15m,1h,4h", help="周期，逗号分隔")
    ap.add_argument("--kline-root", default=str(DEFAULT_KLINE_ROOT), help="kline-data 根目录")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    kline_root = Path(args.kline_root)
    if not kline_root.exists():
        raise SystemExit(f"kline-data 根目录不存在: {kline_root}")

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]

    total = 0
    for tf in timeframes:
        print(f"[{tf}]")
        for sym in symbols:
            try:
                n = import_symbol(args.category, sym, tf, kline_root, verbose=not args.quiet)
                total += n
            except FileNotFoundError as e:
                print(f"  SKIP {sym} {tf}: {e}")
    print(f"DONE total={total} bars")


if __name__ == "__main__":
    main()
