"""BTC 机器学习两连测：
1) 零样本跨市场迁移：A 股存活模型（结构物理）直接打分 BTC，不重新训练。
   —— 若 AUC>0.55，"结构预测结构"是跨市场物理；若≈0.5，存活可学性也是 A 股专属。
2) BTC 原生 WF：1,594 事件 12m 滑动（薄样本，仅指示性）。
特征口径：两组都报——全交集 与 剔除 mkt_/bg_（市场相关特征对单资产 BTC 退化）。
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb

from chan_notes.eval.metrics import auc
from chan_notes.models import train

A = train.load("events_v7s")   # A股 41k 事件，survive_144
B = train.load("events_b1s")   # BTC 1.6k 事件

feat_all = train.feature_cols(A)
feat_all = [c for c in feat_all if c in B.columns]
feat_nomkt = [c for c in feat_all if not c.startswith(("mkt_", "bg_"))]
print(f"特征交集: {len(feat_all)} (剔除 mkt_/bg_ 后 {len(feat_nomkt)})")

# ---------- 1) 零样本迁移 ----------
for tag, feats in (("全交集", feat_all), ("无市场特征", feat_nomkt)):
    Xa = A[feats].fillna(0.0).values
    ya = A.survive_144.astype(int).values
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(Xa))
    cut = int(len(Xa) * 0.9)
    booster = train.fit_xgb(Xa[idx[:cut]], ya[idx[:cut]], Xa[idx[cut:]], ya[idx[cut:]])  # 全量 A 股训练（结构物理模型）
    for side, m in (("BUY", B.is_buy == 1), ("SELL", B.is_buy == 0)):
        sub = B[m]
        p = booster.predict(xgb.DMatrix(sub[feats].fillna(0.0).values))
        y = sub.survive_144.astype(int).values
        print(f"[迁移/{tag}] {side}: n={len(y)} AUC={auc(y, p):.4f}")

# ---------- 2) BTC 原生 WF ----------
import xgboost as xgb
B = B.copy()
B["month"] = pd.to_datetime(B.event_time).dt.to_period("M")
feats = feat_nomkt
months = sorted(B.month.unique())
print(f"\n[BTC 原生 WF] 窗口 12m, min_train=300")
for side, m0 in (("BUY", B.is_buy == 1), ("SELL", B.is_buy == 0)):
    d = B[m0]
    aucs = []
    for i, mo in enumerate(months):
        if i < 12:
            continue
        tr = d[(d.month >= months[i - 12]) & (d.month < mo)]
        te = d[d.month == mo]
        if len(tr) < 300 or len(te) < 5 or te.survive_144.nunique() < 2:
            continue
        Xtr = tr[feats].fillna(0.0).values
        ytr = tr.survive_144.astype(int).values
        cut = max(10, int(len(Xtr) * 0.9))
        bst = train.fit_xgb(Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:])
        p = bst.predict(xgb.DMatrix(te[feats].fillna(0.0).values))
        aucs.append(auc(te.survive_144.astype(int).values, p))
    if aucs:
        a = np.array(aucs)
        print(f"  {side}: {len(a)} 个月, mean={a.mean():.4f} min={a.min():.4f} >=0.55 占比 {(a>=0.55).mean()*100:.0f}%")
    else:
        print(f"  {side}: 样本不足")
