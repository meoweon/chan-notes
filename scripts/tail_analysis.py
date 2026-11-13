"""Analyze the top-5% tail winners: are they regime luck or repeatable structure?

Checks: time clustering, stock clustering, and whether the walk-forward
big-move score (wf_buy_big) ranks tail winners above average.
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
df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
arrays = {c: for_policy(a) for c, a in load_many(df.code.unique()).items()}

rets = []
for r in df.itertuples():
    res = simulate_exit(arrays[r.code], int(r.conf_entry_idx), 1, float(r.atr),
                        hold=HOLD, stop_atr=STOP, trail_atr=0.0)
    rets.append(res.ret if res is not None else np.nan)
df["ret"] = rets
df = df.dropna(subset=["ret"])
df["ym"] = pd.to_datetime(df["event_time"]).dt.to_period("M").astype(str)
df["year"] = pd.to_datetime(df["event_time"]).dt.year

thr = df["ret"].quantile(0.95)
top = df[df["ret"] >= thr]
print(f"total trades {len(df)}, top5% threshold {thr*1e4:.0f}bp, tail n={len(top)}")
print(f"tail mean {top['ret'].mean()*1e4:.0f}bp | rest mean {df.loc[df.ret<thr,'ret'].mean()*1e4:.0f}bp")

# 1. time clustering
by_m = top.groupby("ym").size()
total_by_m = df.groupby("ym").size()
rate = (by_m / total_by_m * 100).fillna(0).sort_index()
print("\ntail share by month (%, months >= 8% shown):")
hot = rate[rate >= 8]
print(hot.round(1).to_string())
print(f"\nmonths with tail share >=8%: {len(hot)} / {len(rate)}")
print("top-10 tail months:", list(rate.sort_values(ascending=False).head(10).index))

# 2. stock clustering
by_code = top.groupby("code").size().sort_values(ascending=False)
print(f"\ntail spread over {by_code.size} stocks; top-10 stocks carry {by_code.head(10).sum()/len(top)*100:.0f}% of tail")
print(by_code.head(10).to_string())

# 3. year distribution
yr = top.groupby("year").size() / df.groupby("year").size() * 100
print("\ntail share by year (%):")
print(yr.round(1).to_string())

# 4. does the WF big-move score identify the tail?
ml_path = Path(__file__).resolve().parents[1] / "data" / "events" / "wf_buy_big.parquet"
if ml_path.exists():
    ml = pd.read_parquet(ml_path)
    print("\nwf_buy_big columns:", list(ml.columns)[:8])
    key = ["code", "event_time"]
    m = df.merge(ml, on=key, how="inner", suffixes=("", "_ml"))
    sc = [c for c in ml.columns if c not in key and c not in df.columns or c.endswith("_ml")]
    scol = [c for c in ml.columns if c not in key][0] if len([c for c in ml.columns if c not in key]) == 1 else None
    if scol is None:
        cand = [c for c in ml.columns if "score" in c or "pred" in c or "prob" in c]
        scol = cand[0] if cand else None
    if scol and len(m) > 100:
        in_tail = m["ret"] >= thr
        from chan_notes.eval.metrics import auc
        a = auc(m[scol].values, in_tail.astype(int).values)
        hi = m[m[scol] >= m[scol].quantile(0.8)]
        lo = m[m[scol] < m[scol].quantile(0.8)]
        print(f"\nscore col = {scol}, merged n={len(m)}")
        print(f"AUC(score -> in top5% tail) = {a:.4f}")
        print(f"tail rate in top20% score: {(hi['ret']>=thr).mean()*100:.1f}% vs rest {(lo['ret']>=thr).mean()*100:.1f}%")
        print(f"mean ret top20% score: {hi['ret'].mean()*1e4:.0f}bp | rest: {lo['ret'].mean()*1e4:.0f}bp")
else:
    print("\n(no wf_buy_big predictions found)")
