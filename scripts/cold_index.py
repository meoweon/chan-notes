"""冷时段（frenzy 关机期）的现金替代：持有 _MARKET 指数 vs 空仓。
对照口径：OOS 2024-2026，门控关闭日持有指数（次日开盘调仓）的累计收益。
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "2024-01-01"

rg = pd.read_csv(ROOT / "reports" / "market_regime.csv", parse_dates=["d"]).set_index("d")
fz = pd.read_csv(ROOT / "reports" / "frenzy_index.csv", parse_dates=["d"]).set_index("d")["frenzy5"].shift(1)
thr = fz[fz.index < SPLIT].median()

mk = pd.read_parquet(ROOT / "data" / "bars" / "d" / "_MARKET.parquet")
mk["time"] = pd.to_datetime(mk["time"])
mk = mk.set_index("time").sort_index()
mk["ret"] = mk["close"].pct_change()
mk["state"] = rg["state"]
mk["fz"] = fz
mk["cold"] = mk["fz"] < thr

oos = mk[mk.index >= SPLIT].copy()
for tag, mask in (
    ("冷时段全部", oos.cold),
    ("冷+range", oos.cold & (oos.state == "range")),
    ("冷+trend_up", oos.cold & (oos.state == "trend_up")),
    ("热时段(对照)", ~oos.cold),
):
    r = oos.ret[mask.fillna(False)]
    cum = (1 + r.fillna(0)).prod() - 1
    days = mask.sum()
    print(f"{tag:<14s}: {days:>4d} 天, 累计 {cum*100:>+6.1f}%, 日均 {r.mean()*1e4:>+5.1f}bp, 日胜率 {(r>0).mean()*100:.0f}%")
print(f"\nOOS 全程买入持有指数: {(1+oos.ret.fillna(0)).prod()-1:+.1%}")
