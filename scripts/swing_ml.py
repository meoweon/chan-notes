"""摆动点事件 ML 终审（用户提案：自定义高低点 + 特征 + 机器学习）。
- 标签：288/3.0 与 576/3.0 政策收益（真实成交口径，成本 12bp）
- 特征：114 个非缠论特征（pa_/pt_/sw_/rg_/bc_/bsw_/ff_/rf_/wg_/bg_/mkt_）
- 验证：12 个月滑动 WF（mcw 按 n 缩放），分买/卖侧报 AUC + top/bottom20
- 附带 IS(2020-23) 随机安慰剂（补全 OOS 已有的对照）
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb

from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many
from chan_notes.eval.metrics import auc as mw_auc

COST = 0.0012
FEAT_PREFIX = ("pa_", "pt_", "sw_", "rg_", "bc_", "bsw_", "ff_", "rf_", "wg_", "bg_", "mkt_")

df = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "events" / "swing_b30_a1.5.parquet")
b = for_policy(load_many(["BTCUSDT"], "30m")["BTCUSDT"])
c, h, l = b["close"], b["high"], b["low"]
pc = np.roll(c, 1); pc[0] = c[0]
tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
b["atr"] = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()

# ---- IS 安慰剂（补全）----
rng = np.random.default_rng(11)
times = pd.to_datetime(pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "bars_crypto" / "30m/BTCUSDT.parquet")["time"]).to_numpy()
is_idx = np.where(times < np.datetime64("2024-01-01"))[0]
lo, hi = int(is_idx.min()), int(is_idx.max()) - 700
for hold in (288, 576):
    rnd = []
    for i in rng.integers(lo, hi, size=1500):
        res = simulate_exit(b, int(i), 0, float(b["atr"][i]), hold=hold, stop_atr=3.0, cost=COST)
        if res is not None:
            rnd.append(res.ret)
    print(f"IS 随机做空 hold={hold}: E={np.mean(rnd)*1e4:+.1f}bp  (对照信号 IS E=+19.2/+37.9)")

# ---- 标签 ----
for hold in (288, 576):
    col = f"sw{hold}_ret"
    if col not in df.columns:
        rets = []
        for r in df.itertuples():
            res = simulate_exit(b, int(r.conf_entry_idx), int(r.is_buy), float(r.atr), hold=hold, stop_atr=3.0, cost=COST)
            rets.append(res.ret if res is not None else np.nan)
        df[col] = rets
df["month"] = df.event_time.str[:7]
df = df.dropna(subset=["sw288_ret", "sw576_ret"]).reset_index(drop=True)

feats = [c0 for c0 in df.columns if c0.startswith(FEAT_PREFIX)]
print(f"\nfeatures={len(feats)}  events={len(df)}")

# ---- 12m 滑动 WF（mcw 按 n 缩放）----
def wf(sub, label):
    sub = sub.sort_values("event_time").reset_index(drop=True)
    months = sorted(sub.month.unique())
    preds = []
    for i, m in enumerate(months):
        if i < 12:
            continue
        win = months[i - 12:i]
        tr_df = sub[sub.month.isin(win)]
        te = sub[sub.month == months[i]]
        if len(tr_df) < 300 or len(te) < 5 or te[label].gt(0).nunique() < 2:
            continue
        n = len(tr_df)
        params = {"max_depth": 3, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
                  "min_child_weight": max(1, min(20, int(n * 0.05))), "objective": "binary:logistic"}
        nva = max(20, int(n * 0.1))
        dtr = xgb.DMatrix(tr_df[feats].iloc[:-nva], label=(tr_df[label].iloc[:-nva] > 0).astype(int))
        dva = xgb.DMatrix(tr_df[feats].iloc[-nva:], label=(tr_df[label].iloc[-nva:] > 0).astype(int))
        mdl = xgb.train(params, dtr, 120, evals=[(dva, "v")], early_stopping_rounds=15, verbose_eval=False)
        s = mdl.predict(xgb.DMatrix(te[feats]))
        preds.append(pd.DataFrame({"month": months[i], "y": (te[label] > 0).astype(int).to_numpy(),
                                   "ret": te[label].to_numpy(), "score": s}))
    return pd.concat(preds) if preds else pd.DataFrame()

for side, lbl in ((1, "BUY"), (0, "SELL-short")):
    sub = df[df.is_buy == side]
    for hold in (288, 576):
        oof = wf(sub, f"sw{hold}_ret")
        if oof.empty:
            print(f"{lbl} {hold}: no oof"); continue
        a = mw_auc(oof.y.to_numpy(), oof.score.to_numpy())
        q80 = oof.score.quantile(0.8); q20 = oof.score.quantile(0.2)
        top = oof[oof.score >= q80].ret.mean() * 1e4
        bot = oof[oof.score <= q20].ret.mean() * 1e4
        base = oof.ret.mean() * 1e4
        # 逐月稳定性
        mo = oof.groupby("month").apply(lambda g: mw_auc(g.y.to_numpy(), g.score.to_numpy()) if g.y.nunique() > 1 else np.nan).dropna()
        print(f"{lbl:10s} hold={hold}: oof_n={len(oof)} AUC={a:.4f} | top20={top:+.1f} base={base:+.1f} bot20={bot:+.1f}bp | "
              f"月均AUC={mo.mean():.3f} ≥0.55占{(mo >= 0.55).mean() * 100:.0f}% ({len(mo)}月)")
