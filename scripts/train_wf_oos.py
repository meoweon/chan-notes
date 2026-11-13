"""Walk-forward 训练，并落盘**逐事件样本外预测**。

`train_wf_xgb.py` 只保存最后一个 fold 的模型；用它给全历史打分会把未来信息
带回过去（2020 年的事件由 2026-09 训练的模型评分）。本脚本逐月 walk-forward，
把每个月测试集的预测按事件行号存下来，回测端按行号取分即可保证样本外。

用法：
  python scripts/train_wf_oos.py [dataset] [out] [label_col] [subset]
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from chan_notes.config import MODELS
from chan_notes.features import schema
from chan_notes.models import train


def _subset(df, name):
    if name == "buy":
        return df[df.is_buy == 1]
    if name == "sell":
        return df[df.is_buy == 0]
    if name == "conf_buy":
        return df[(df.is_buy == 1) & (df.conf_ok == 1)]
    if name == "conf_all":
        return df[df.conf_ok == 1]
    return df


def run(name="crypto_21_v1", out="wf_crypto_21_xgb_oos", label="survive_144",
        subset="all", min_train=500, matures_days=28):
    raw = _subset(train.load(name), subset).copy()
    # 保留原始行号：回测端用同一个 parquet 的行号取分
    raw["__idx"] = raw.index
    raw["t"] = pd.to_datetime(raw["event_time"])
    df = raw.sort_values("t").reset_index(drop=True)
    df = df[pd.notna(df[label])]
    y = df[label].astype(int).to_numpy()
    cols = [c for c in schema.load() if c in df.columns]
    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"events={len(df)} pos={y.mean():.3f} features={len(cols)}")

    months = pd.period_range(df["t"].min().to_period("M") + 12,
                             df["t"].max().to_period("M"), freq="M")
    preds, oos_idx, oos_prob, oos_month, last = [], [], [], [], None

    xgb_params = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }

    for m in months:
        start = m.start_time
        cut = start - pd.Timedelta(days=matures_days)
        tr = df["t"] <= cut
        nxt = (m + 1).start_time
        te = (df["t"] >= start) & (df["t"] < nxt)
        if tr.sum() < min_train or te.sum() < 5:
            continue
        X_tr, y_tr = X.loc[tr], y[tr]
        X_te, y_te = X.loc[te], y[te]

        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, prob) if len(np.unique(y_te)) > 1 else np.nan
        preds.append({"month": str(m), "auc": auc, "n_test": int(te.sum())})
        oos_idx.extend(df.loc[te, "__idx"].to_numpy())
        oos_prob.extend(prob)
        oos_month.extend([str(m)] * int(te.sum()))
        last = clf
        print(f"  {m}: AUC={auc:.4f} n_test={te.sum()}")

    a = pd.DataFrame(preds)
    print(f"\n逐月 AUC: mean={a.auc.mean():.4f}  std={a.auc.std():.4f}  min={a.auc.min():.4f}  "
          f">=0.55 占比 {(a.auc >= 0.55).mean()*100:.0f}%  ({len(a)} 个月)")

    out_path = MODELS / f"{out}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({
            "model": last, "features": cols, "meta": a.to_dict("list"),
            "oos_idx": np.asarray(oos_idx, dtype=np.int64),
            "oos_prob": np.asarray(oos_prob, dtype=float),
            "oos_month": oos_month,
            "label": label, "subset": subset,
        }, f)
    print(f"saved {out_path}  oos_events={len(oos_idx)}")
    return a


if __name__ == "__main__":
    argv = sys.argv[1:]
    name = argv[0] if len(argv) > 0 else "crypto_21_v1"
    out = argv[1] if len(argv) > 1 else "wf_crypto_21_xgb_oos"
    label = argv[2] if len(argv) > 2 else "survive_144"
    subset = argv[3] if len(argv) > 3 else "all"
    run(name, out, label, subset)
