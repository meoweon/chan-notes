"""BTC 原生训练检验（按用户提议的 train/test 流程）：
A) 存活标签（n=1,594）：12m 滑动窗口 min_train=100 + 时间切分 IS(2020-23)/OOS(2024-26)
B) 收益标签（conf_buy n=263）：IS 训练 / OOS 测试，报 AUC 与 bootstrap CI
——目的：量化"样本不足"到底是多少不足，而不是断言。
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

df = train.load("events_b1")
feats = [c for c in train.feature_cols(df) if not c.startswith(("mkt_", "bg_"))]
df["month"] = pd.to_datetime(df.event_time).dt.to_period("M")
df["year"] = pd.to_datetime(df.event_time).dt.year
print(f"BTC 事件: {len(df)} (BUY {(df.is_buy==1).sum()} / SELL {(df.is_buy==0).sum()}), "
      f"{df.month.nunique()} 个月, 月均 {len(df)/df.month.nunique():.0f} 事件")

def fit(tr, te, ycol, feats):
    Xtr = tr[feats].fillna(0.0).values; ytr = tr[ycol].astype(int).values
    Xte = te[feats].fillna(0.0).values
    n = len(ytr)
    # min_child_weight 按样本量缩放：mcw=20 在 n<800 时会瘫痪树（子节点 hessian 永远不够）
    mcw = max(1.0, min(20.0, n * 0.05))
    params = dict(train.XGB_PARAMS, min_child_weight=mcw)
    pos = np.where(ytr == 1)[0]; neg = np.where(ytr == 0)[0]
    rng = np.random.RandomState(42)
    va = np.concatenate([rng.choice(pos, max(1, len(pos)//10), replace=False),
                         rng.choice(neg, max(1, len(neg)//10), replace=False)])
    mask = np.ones(n, bool); mask[va] = False
    if len(np.unique(ytr[va])) < 2 or len(Xtr[mask]) < 50:
        bst = xgb.train(params, xgb.DMatrix(Xtr, label=ytr), num_boost_round=100)
    else:
        bst = train.fit_xgb(Xtr[mask], ytr[mask], Xtr[va], ytr[va], params=params)
    return bst.predict(xgb.DMatrix(Xte))

def boot_ci(y, p, n=2000, seed=0):
    rng = np.random.RandomState(seed); vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(np.unique(y[i])) < 2: continue
        vals.append(auc(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])

# ---------- A) 存活 ----------
print("\n=== A) 存活标签 survive_144（BTC 原生）===")
ds = df.dropna(subset=["survive_144"])
# 12m 滑动 min_train=100
for side, m0 in (("BUY", ds.is_buy == 1), ("SELL", ds.is_buy == 0)):
    d = ds[m0]; months = sorted(d.month.unique()); aucs = []
    for i, mo in enumerate(months):
        if i < 12: continue
        tr = d[(d.month >= months[i-12]) & (d.month < mo)]; te = d[d.month == mo]
        if len(tr) < 100 or len(te) < 5 or te.survive_144.nunique() < 2: continue
        p = fit(tr, te, "survive_144", feats)
        aucs.append((str(mo), auc(te.survive_144.astype(int).values, p)))
    if aucs:
        a = np.array([x[1] for x in aucs])
        print(f"  {side} 12m滑动(min100): {len(a)} 个月, mean={a.mean():.4f} min={a.min():.4f} >=0.55占{(a>=0.55).mean()*100:.0f}%")
# 时间切分
for side, m0 in (("BUY", ds.is_buy == 1), ("SELL", ds.is_buy == 0)):
    d = ds[m0]
    tr = d[d.year <= 2023]; te = d[d.year >= 2024]
    p = fit(tr, te, "survive_144", feats)
    a = auc(te.survive_144.astype(int).values, p)
    lo, hi = boot_ci(te.survive_144.astype(int).values, p)
    print(f"  {side} IS训练→OOS测试: n_tr={len(tr)} n_te={len(te)} AUC={a:.4f} CI95=[{lo:.3f},{hi:.3f}]")

# ---------- B) 收益 ----------
print("\n=== B) 收益标签 pol144_ret>0（conf_buy, BTC 原生）===")
cb = df[(df.is_buy == 1) & (df.conf_ok == 1) & df.pol144_ret.notna()].copy()
cb["y"] = (cb.pol144_ret > 0).astype(int)
print(f"  可用样本 n={len(cb)}（IS {(cb.year<=2023).sum()} / OOS {(cb.year>=2024).sum()}），"
      f"事件/特征比 = {len(cb)}/{len(feats)} = {len(cb)/len(feats):.1f}")
tr = cb[cb.year <= 2023]; te = cb[cb.year >= 2024]
p = fit(tr, te, "y", feats)
a = auc(te.y.values, p); lo, hi = boot_ci(te.y.values, p)
print(f"  IS训练→OOS测试: n_tr={len(tr)} n_te={len(te)} AUC={a:.4f} CI95=[{lo:.3f},{hi:.3f}]")
# 高分组的实际收益（学习到的排序有没有钱）
te = te.copy(); te["score"] = p
q = te.score.quantile(0.8)
hi_g = te[te.score >= q]; lo_g = te[te.score < q]
print(f"  OOS top20%: n={len(hi_g)} E={hi_g.pol144_ret.mean()*1e4:+.1f}bp | 其余: n={len(lo_g)} E={lo_g.pol144_ret.mean()*1e4:+.1f}bp")
# 5 折交叉验证（用户提议的验证集做法，看稳定性）
from numpy.random import RandomState
rng = RandomState(7); idx = rng.permutation(len(cb)); folds = np.array_split(idx, 5)
cv = []
for k in range(5):
    te_i = folds[k]; tr_i = np.concatenate([folds[j] for j in range(5) if j != k])
    p = fit(cb.iloc[tr_i], cb.iloc[te_i], "y", feats)
    if cb.iloc[te_i].y.nunique() > 1:
        cv.append(auc(cb.iloc[te_i].y.values, p))
print(f"  5折CV AUC: {[f'{x:.3f}' for x in cv]} mean={np.mean(cv):.4f}")
