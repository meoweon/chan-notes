"""Regime gate test: condition 144/3.0 expectancy on market-state features at entry.

If the tail is regime-gated, E should be strongly positive only in the top
quantiles of speculative-regime proxies (mkt_ret20, mkt_ma20_slope, ...).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many
from chan_notes.eval.metrics import expectancy
from chan_notes.models import train

HOLD, STOP = 144, 3.0
GATE_FEATS = ["mkt_ret20", "mkt_ret5", "mkt_ma20_slope", "mkt_atr_pct",
              "pa_atr_pct", "pa_vol_ratio20", "mkt_dist_ma20", "mkt_pos_in_range20"]

df = train.load("events_v4")
df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
arrays = {c: for_policy(a) for c, a in load_many(df.code.unique()).items()}

rets = []
for r in df.itertuples():
    res = simulate_exit(arrays[r.code], int(r.conf_entry_idx), 1, float(r.atr),
                        hold=HOLD, stop_atr=STOP, trail_atr=0.0)
    rets.append(res.ret if res is not None else np.nan)
df["ret"] = rets
df = df.dropna(subset=["ret"])

avail = [f for f in GATE_FEATS if f in df.columns]
print("available gate features:", avail)
rows = []
for f in avail:
    d = df.dropna(subset=[f])
    try:
        q = pd.qcut(d[f], 4, labels=False, duplicates="drop")
    except ValueError:
        continue
    for qi in sorted(q.unique()):
        sub = d[q == qi]
        e = expectancy(sub["ret"].values)
        rows.append({"feature": f, "quartile": qi + 1, "n": e["n"],
                     "E_bp": e["expectancy_bp"], "P": e["win_rate"], "R": e["payoff_R"],
                     "feat_lo": float(sub[f].min()), "feat_hi": float(sub[f].max())})
out = pd.DataFrame(rows)
print()
print(out.round(3).to_string(index=False))
out.to_csv(Path(__file__).resolve().parents[1] / "reports" / "regime_gate.csv", index=False)
print("\nsaved reports/regime_gate.csv")
