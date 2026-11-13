"""regime×期望 对账表：conf_buy（events_v8, pol144_ret）按市场状态(trend_up/down/range)×frenzy(热/冷)切分。
PIT：事件日 T 用 T-1 日的市场状态与 frenzy5。
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
fz = pd.read_csv(ROOT / "reports" / "frenzy_index.csv", parse_dates=["d"]).set_index("d")["frenzy5"]
thr = fz[fz.index < SPLIT].median()

df = pd.read_parquet(ROOT / "data" / "events" / "events_v8.parquet")
df = df[(df.is_buy == 1) & (df.conf_ok == 1) & df.pol144_ret.notna()].copy()
df["d"] = pd.to_datetime(df["event_time"]).dt.normalize()
prev = rg.shift(1)
df["state"] = df["d"].map(prev["state"])
df["fz"] = df["d"].map(fz.shift(1))
df = df.dropna(subset=["state", "fz"])
df["heat"] = np.where(df.fz >= thr, "热", "冷")
df["istail"] = df.pol144_ret > 0.30

print(f"conf_buy n={len(df)}")
print(f"\n{'状态':<10s}{'热度':<4s}{'n':>6s}{'E(bp)':>9s}{'胜率':>7s}{'尾部率':>8s}{'IS E':>9s}{'OOS E':>9s}")
for st in ("trend_up", "trend_down", "range"):
    for ht in ("热", "冷"):
        s = df[(df.state == st) & (df.heat == ht)]
        if len(s) < 10:
            print(f"{st:<10s}{ht:<4s}{len(s):>6d}   (样本不足)")
            continue
        e = s.pol144_ret.mean() * 1e4
        w = (s.pol144_ret > 0).mean() * 100
        t = s.istail.mean() * 100
        e_is = s[s.event_time < SPLIT].pol144_ret.mean() * 1e4
        e_oos = s[s.event_time >= SPLIT].pol144_ret.mean() * 1e4
        print(f"{st:<10s}{ht:<4s}{len(s):>6d}{e:>+9.1f}{w:>6.1f}%{t:>7.2f}%{e_is:>+9.1f}{e_oos:>+9.1f}")

print("\n附：ER60 连续口径（全样本五分位）")
df["er"] = df["d"].map(prev["er60"])
q = pd.qcut(df.er, 5, labels=False, duplicates="drop")
for i in range(5):
    s = df[q == i]
    print(f"  ER Q{i+1} [{s.er.min():.3f},{s.er.max():.3f}]: n={len(s):5d} E={s.pol144_ret.mean()*1e4:+7.1f}bp 尾部率={s.istail.mean()*100:.2f}%")
