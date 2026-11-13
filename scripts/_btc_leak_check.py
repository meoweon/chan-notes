"""BTC 卖侧 0.99 AUC 泄漏诊断：单变量 AUC 排序 + 头部特征定义核查。"""
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["CHAN_NOTES_BARS"] = str(Path(__file__).resolve().parents[1] / "data" / "bars_crypto")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many
from chan_notes.eval.metrics import auc
from chan_notes.models import train

df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "events" / "swing_b30_a1.5.parquet")
df["event_time"] = pd.to_datetime(df["event_time"])
b = for_policy(load_many(["BTCUSDT"], "30m")["BTCUSDT"])

sub = df[df.is_buy == 0].copy()
rets = np.full(len(sub), np.nan)
for i, r in enumerate(sub.itertuples()):
    res = simulate_exit(b, int(r.conf_entry_idx), 0, float(r.atr), hold=576, stop_atr=3.0, cost=0.0012)
    if res is not None:
        rets[i] = res.ret
sub["ret576"] = rets
sub = sub.dropna(subset=["ret576"])
thr = np.quantile(sub.ret576.to_numpy(), 0.98)
y = (sub.ret576.to_numpy() > thr).astype(int)
print(f"n={len(sub)} thr={thr*1e2:+.2f}% pos={y.sum()}")

keep = [c for c in train.feature_cols(sub) if c in sub.columns]
rows = []
for c in keep:
    v = sub[c].to_numpy(dtype=float)
    if np.isnan(v).all() or np.nanstd(v) == 0:
        continue
    m = ~np.isnan(v)
    rows.append((c, auc(y[m], v[m])))
rows.sort(key=lambda x: -abs(x[1] - 0.5))
for c, a in rows[:12]:
    print(f"{c:24s} univ AUC={a:.4f}")
