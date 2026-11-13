"""Augment events_v4 with the REAL trade label: 144bar/3ATR policy return
from the confirmation entry (conf_entry_idx). Saves events_v5.

The framework: chan BSP -> chan+PA features -> sliding-window ML scores
reliability of the actual trade we would take.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many
from chan_notes.models import train

HOLD, STOP = 144, 3.0

df = train.load("events_v4")
arrays = {c: for_policy(a) for c, a in load_many(df.code.unique()).items()}

rets = []
for r in df.itertuples():
    if r.is_buy == 1 and r.conf_ok == 1:
        res = simulate_exit(arrays[r.code], int(r.conf_entry_idx), 1, float(r.atr),
                            hold=HOLD, stop_atr=STOP, trail_atr=0.0)
        rets.append(res.ret if res is not None else np.nan)
    else:
        rets.append(np.nan)
df["pol144_ret"] = rets

out = Path(__file__).resolve().parents[1] / "data" / "events" / "events_v5.parquet"
df.to_parquet(out, index=False)
ok = df.dropna(subset=["pol144_ret"])
print(f"saved {out} rows={len(df)}; labeled conf_buy={len(ok)} "
      f"mean={ok['pol144_ret'].mean()*1e4:.1f}bp win={(ok['pol144_ret']>0).mean()*100:.1f}%")
