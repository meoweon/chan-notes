"""把 30m K 线重采样为 UTC 日线（加密 24/7，日界=UTC 00:00）。

用法：set CHAN_NOTES_BARS=<bars_root> && python scripts/resample_daily.py
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import pandas as pd

from chan_notes.data import store


def main():
    codes = [c for c in store.list_codes("30m") if not c.startswith("_")]
    for c in codes:
        df = store.load(c, "30m")
        g = df.set_index("time").resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        g = g.dropna(subset=["close"]).reset_index()
        store.save(c, g, "d")
        print(c, len(g), g.time.iloc[0], g.time.iloc[-1], flush=True)
    print("DONE", len(codes))


if __name__ == "__main__":
    main()
