"""Quantify regime dependence of the tail: how much of the P&L is rally beta?"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many
from chan_notes.models import train

HOLD, STOP = 144, 3.0
RALLY_MONTHS = {"2020-06", "2020-07", "2024-09", "2024-10"}

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

total_pnl = df["ret"].sum()
df["rally"] = df["ym"].isin(RALLY_MONTHS)
print(f"total net P&L (sum of trade rets): {total_pnl*1e4:.0f}bp over {len(df)} trades")
for flag, grp in df.groupby("rally"):
    print(f"  rally_window={flag}: n={len(grp)} ({len(grp)/len(df)*100:.0f}%), "
          f"mean {grp['ret'].mean()*1e4:.0f}bp, P&L share {grp['ret'].sum()/total_pnl*100:.0f}%")

# market forward return over the holding window for each trade
mkt = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "bars" / "30m" / "_MARKET.parquet")
print("\n_MARKET columns:", list(mkt.columns))
close_col = "close" if "close" in mkt.columns else mkt.columns[-1]
mc = mkt[close_col].values
times = mkt["time"].astype(str).values if "time" in mkt.columns else mkt.index.astype(str)
t2i = {t: i for i, t in enumerate(times)}

et = pd.to_datetime(df["event_time"]).dt.strftime("%Y%m%d%H%M%S")
mkt_fwd = []
for t in et:
    cands = [i for k, i in t2i.items() if k[:12] <= t[:12]]
    i = max(cands) if cands else None
    if i is None or i + HOLD >= len(mc):
        mkt_fwd.append(np.nan)
    else:
        mkt_fwd.append(mc[i + HOLD] / mc[i] - 1)
df["mkt_fwd"] = mkt_fwd
ok = df.dropna(subset=["mkt_fwd"])
r = np.corrcoef(ok["ret"], ok["mkt_fwd"])[0, 1]
print(f"\ncorr(trade ret, market fwd {HOLD}bar ret) = {r:.3f}  (n={len(ok)})")
thr = df["ret"].quantile(0.95)
tail = ok[ok["ret"] >= thr]
rest = ok[ok["ret"] < thr]
print(f"tail mean mkt_fwd: {tail['mkt_fwd'].mean()*1e4:.0f}bp | rest mean mkt_fwd: {rest['mkt_fwd'].mean()*1e4:.0f}bp")
q = ok["mkt_fwd"].quantile(0.8)
hot = ok[ok["mkt_fwd"] >= q]
cold = ok[ok["mkt_fwd"] < q]
print(f"trades with market fwd in top20%: n={len(hot)}, mean ret {hot['ret'].mean()*1e4:.0f}bp, "
      f"tail rate {(hot['ret']>=thr).mean()*100:.1f}%")
print(f"trades with market fwd in bot80%: n={len(cold)}, mean ret {cold['ret'].mean()*1e4:.0f}bp, "
      f"tail rate {(cold['ret']>=thr).mean()*100:.1f}%")
