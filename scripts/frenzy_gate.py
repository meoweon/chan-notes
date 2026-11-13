"""Speculation-index regime gate: build a daily 'frenzy' proxy from the universe
itself (share of stocks with daily gain > 9.5%, 5-day smoothed), then condition
144/3.0 expectancy on its quantile at entry.
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

ROOT = Path(__file__).resolve().parents[1]
BAR_DIR = ROOT / "data" / "bars" / "30m"

# --- build daily frenzy index from 30m bars of all stocks ---
daily = {}
for f in sorted(BAR_DIR.glob("*.parquet")):
    if f.stem == "_MARKET":
        continue
    b = pd.read_parquet(f, columns=["time", "close"])
    b["d"] = b["time"].astype(str).str[:10]
    g = b.groupby("d")["close"].last()
    daily[f.stem] = g
px = pd.DataFrame(daily).sort_index()
rets = px.pct_change()
frenzy = (rets > 0.095).sum(axis=1) / rets.notna().sum(axis=1)
frenzy5 = frenzy.rolling(5, min_periods=1).mean()
print(f"daily frenzy index: {frenzy.index[0]}..{frenzy.index[-1]}, "
      f"mean {frenzy.mean()*100:.2f}%, p95 {frenzy.quantile(0.95)*100:.2f}%")
print("top-5 frenzy days:", list(frenzy.sort_values(ascending=False).head(5).index))

# --- simulate trades ---
df = train.load("events_v4")
df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
arrays = {c: for_policy(a) for c, a in load_many(df.code.unique()).items()}
rets_l = []
for r in df.itertuples():
    res = simulate_exit(arrays[r.code], int(r.conf_entry_idx), 1, float(r.atr),
                        hold=HOLD, stop_atr=STOP, trail_atr=0.0)
    rets_l.append(res.ret if res is not None else np.nan)
df["ret"] = rets_l
df = df.dropna(subset=["ret"])

df["d"] = pd.to_datetime(df["event_time"]).dt.strftime("%Y-%m-%d")
df["frenzy"] = df["d"].map(frenzy5)
df = df.dropna(subset=["frenzy"])

thr95 = df["ret"].quantile(0.95)
q = pd.qcut(df["frenzy"], 4, labels=False, duplicates="drop")
rows = []
for qi in sorted(q.unique()):
    sub = df[q == qi]
    e = expectancy(sub["ret"].values)
    rows.append({"quartile": qi + 1, "n": e["n"], "E_bp": e["expectancy_bp"],
                 "P": e["win_rate"], "R": e["payoff_R"],
                 "tail_rate_%": float((sub["ret"] >= thr95).mean() * 100),
                 "frenzy_lo": float(sub["frenzy"].min()), "frenzy_hi": float(sub["frenzy"].max())})
out = pd.DataFrame(rows)
print()
print(out.round(3).to_string(index=False))

# binary gate: trade only when frenzy5 above median / above p75
for name, m in [("frenzy >= median", df["frenzy"] >= df["frenzy"].median()),
                ("frenzy >= p75", df["frenzy"] >= df["frenzy"].quantile(0.75)),
                ("frenzy < p75", df["frenzy"] < df["frenzy"].quantile(0.75))]:
    sub = df[m]
    e = expectancy(sub["ret"].values)
    print(f"{name}: n={e['n']} E={e['expectancy_bp']:.1f}bp P={e['win_rate']:.3f} R={e['payoff_R']:.2f}")

out.to_csv(ROOT / "reports" / "frenzy_gate.csv", index=False)
frenzy.to_frame("frenzy").join(frenzy5.rename("frenzy5")).to_csv(ROOT / "reports" / "frenzy_index.csv")
print("\nsaved reports/frenzy_gate.csv + reports/frenzy_index.csv")
