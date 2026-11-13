"""尾部标签的 ATR 归一化对照：z = ret / atr_pct，尾部 = z > 98 分位。
若固定%阈值的"可预测性"是波动率缩放伪影，归一化后 pa_atr_pct 单变量与 WF AUC 应塌缩。
同时重测 A股 events_v8（#33 标签 +30% → ATR 归一化）。"""
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
from chan_notes.eval.bars import for_policy, load_frames, load_many
from chan_notes.eval.metrics import auc
from chan_notes.models import train


def wf(df, y, keep, label):
    df = df.copy()
    df["y"] = y
    df["month"] = pd.to_datetime(df["event_time"]).dt.to_period("M")
    oof = []
    for m in sorted(df.month.unique()):
        tr = (df.month < m) & (df.month >= m - 12)
        te = df.month == m
        if tr.sum() < 500 or te.sum() < 5:
            continue
        ytr = df.y[tr].to_numpy()
        if ytr.mean() < 0.005 or ytr.mean() > 0.995:
            continue
        spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
        mdl = train.fit_xgb(df.loc[tr, keep], ytr, df.loc[tr, keep], ytr,
                            params={"min_child_weight": 2, "scale_pos_weight": float(spw)}, rounds=120)
        import xgboost as xgb
        p = mdl.predict(xgb.DMatrix(df.loc[te, keep]))
        t = df.loc[te, ["month", "y"]].copy()
        t["score"] = p
        oof.append(t)
    if not oof:
        print(f"{label}: no predictions")
        return
    o = pd.concat(oof)
    a = auc(o.y.to_numpy(), o.score.to_numpy())
    mo = o.groupby("month").apply(lambda g: auc(g.y.to_numpy(), g.score.to_numpy()) if g.y.nunique() > 1 else np.nan, include_groups=False).dropna()
    q = pd.qcut(o.score, 5, labels=False, duplicates="drop")
    top = o[q == q.max()]
    bot = o[q == q.min()]
    print(f"{label}: OOF n={len(o)} AUC={a:.4f} | top20 hit {top.y.mean()*100:.2f}% vs bot20 {bot.y.mean()*100:.2f}% | 逐月 mean={mo.mean():.3f} >=0.55占{(mo>=0.55).mean()*100:.0f}%")


def univ(df, y, keep, feats=("pa_atr_pct", "pa_range_width20", "pa_dist_hh20")):
    out = []
    for c in feats:
        if c in keep:
            v = df[c].to_numpy(dtype=float)
            m = ~np.isnan(v)
            out.append(f"{c}={auc(y[m], v[m]):.4f}")
    return " ".join(out)


# ---- BTC 摆动点 ----
df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "events" / "swing_b30_a1.5.parquet")
df["event_time"] = pd.to_datetime(df["event_time"])
b = for_policy(load_many(["BTCUSDT"], "30m")["BTCUSDT"])
rets = np.full(len(df), np.nan)
for i, r in enumerate(df.itertuples()):
    res = simulate_exit(b, int(r.conf_entry_idx), int(r.is_buy), float(r.atr), hold=576, stop_atr=3.0, cost=0.0012)
    if res is not None:
        rets[i] = res.ret
df["ret576"] = rets

for side, tag in ((1, "BTC BUY"), (0, "BTC SELL")):
    sub = df[(df.is_buy == side) & df.ret576.notna()].copy()
    atr_pct = sub["atr"].to_numpy() / sub["entry_price"].to_numpy() if "entry_price" in sub.columns else None
    if atr_pct is None:
        # atr_pct 特征已在表里
        atr_pct = sub["pa_atr_pct"].to_numpy()
    z = sub.ret576.to_numpy() / atr_pct
    thr = np.quantile(z, 0.98)
    y = (z > thr).astype(int)
    keep = [c for c in train.feature_cols(sub) if c in sub.columns]
    print(f"{tag}: n={len(sub)} z-threshold={thr:.2f}xATR pos={y.mean()*100:.2f}% | univ: {univ(sub, y, keep)}")
    wf(sub, y, keep, f"{tag} ATR归一化尾部 WF")

# ---- A股 events_v8 conf_buy ----
os.environ.pop("CHAN_NOTES_BARS", None)
import importlib
import chan_notes.data.store as store_mod
importlib.reload(store_mod)
import chan_notes.eval.bars as bars_mod
importlib.reload(bars_mod)

a = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "events" / "events_v8.parquet")
a = a[(a.is_buy == 1) & (a.conf_ok == 1) & a.pol144_ret.notna()].copy()
bars = bars_mod.load_frames(a.code.unique())
# pol144_ret 已含成本；用 pa_atr_pct 归一化
z = a["pol144_ret"].to_numpy() / a["pa_atr_pct"].to_numpy()
thr = np.quantile(z, 0.98)
y = (z > thr).astype(int)
keep = [c for c in train.feature_cols(a) if c in a.columns]
print(f"A股 conf_buy: n={len(a)} z-threshold={thr:.2f}xATR pos={y.mean()*100:.2f}% | univ: {univ(a, y, keep)}")
wf(a, y, keep, "A股 ATR归一化尾部 WF")
