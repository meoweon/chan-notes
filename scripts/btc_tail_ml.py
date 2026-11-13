"""BTC 尾部标签 ML：摆动点事件，标签=576bar/3ATR 收益 > 98 分位（对齐 A股 #33 口径）。
对比问题：A股 tail AUC 0.588，BTC 的尾部是否同样可学？
参数：mcw=2 + scale_pos_weight=neg/pos（#28/#33 解瘫痪口径）。
"""
import io
import os
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["CHAN_NOTES_BARS"] = str(Path(__file__).resolve().parents[1] / "data" / "bars_crypto")  # store.py 在 import 时读取
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many
from chan_notes.eval.metrics import auc
from chan_notes.models import train

COST = 0.0012
HOLD, STOP = 576, 3.0

df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "events" / "swing_b30_a1.5.parquet")
df["event_time"] = pd.to_datetime(df["event_time"])
df["month"] = df["event_time"].dt.to_period("M")

b = for_policy(load_many(["BTCUSDT"], "30m")["BTCUSDT"])

rets = np.full(len(df), np.nan)
for i, r in enumerate(df.itertuples()):
    res = simulate_exit(b, int(r.conf_entry_idx), int(r.is_buy), float(r.atr), hold=HOLD, stop_atr=STOP, cost=COST)
    if res is not None:
        rets[i] = res.ret
df["ret576"] = rets

for side, label in ((1, "BUY-long"), (0, "SELL-short")):
    sub = df[(df.is_buy == side) & df.ret576.notna()].copy()
    r = sub.ret576.to_numpy()
    thr = np.quantile(r, 0.98)
    y = (r > thr).astype(int)
    print(f"\n===== {label} n={len(sub)} 尾部阈值 {thr*1e2:+.1f}% 基率 {y.mean()*100:.2f}% =====")
    sub["y"] = y

    keep = [c for c in train.feature_cols(sub) if c in sub.columns]
    months = sorted(sub.month.unique())
    oof = []
    for m in months:
        tr = (sub.month < m) & (sub.month >= m - 12)
        te = sub.month == m
        if tr.sum() < 500 or te.sum() < 5:
            continue
        ytr = sub.y[tr].to_numpy()
        if ytr.mean() < 0.005 or ytr.mean() > 0.995:
            continue
        spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        mdl = train.fit_xgb(sub.loc[tr, keep], ytr, sub.loc[tr, keep], ytr,
                            params={"min_child_weight": 2, "scale_pos_weight": float(spw)}, rounds=120)
        import xgboost as xgb
        p = mdl.predict(xgb.DMatrix(sub.loc[te, keep]))
        t = sub.loc[te, ["code", "event_time", "month", "ret576", "y"]].copy()
        t["score"] = p
        oof.append(t)
    if not oof:
        print("no predictions")
        continue
    o = pd.concat(oof)
    a = auc(o.y.to_numpy(), o.score.to_numpy())
    q = pd.qcut(o.score, 5, labels=False, duplicates="drop")
    top = o[q == q.max()]
    bot = o[q == q.min()]
    print(f"OOF n={len(o)} AUC={a:.4f}")
    print(f"top20: 中奖率 {top.y.mean()*100:.2f}% 均值 {top.ret576.mean()*1e4:+.1f}bp | bot20: {bot.y.mean()*100:.2f}% {bot.ret576.mean()*1e4:+.1f}bp")
    for qq in sorted(o.assign(q=q).groupby("q").groups):
        g = o[q == qq]
        print(f"  Q{qq+1}: hit {g.y.mean()*100:5.2f}%  ret {g.ret576.mean()*1e4:+7.1f}bp")
    mo = o.groupby("month").apply(lambda g: auc(g.y.to_numpy(), g.score.to_numpy()) if g.y.nunique() > 1 else np.nan, include_groups=False).dropna()
    print(f"逐月 AUC: mean={mo.mean():.4f} std={mo.std():.4f} min={mo.min():.4f} >=0.55 占比 {(mo >= 0.55).mean()*100:.0f}% ({len(mo)} 个月)")
