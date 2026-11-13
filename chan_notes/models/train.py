"""模型训练的最小工具集：加载数据集、取特征列、训练 XGBoost。

评估指标在 chan_notes.eval.metrics，训练流程在 scripts/train_wf.py。
"""
import pandas as pd

from chan_notes.config import MODELS
from chan_notes.eval.metrics import auc
from chan_notes.features import schema

XGB_PARAMS = {
    "max_depth": 3, "eta": 0.05, "objective": "binary:logistic",
    "eval_metric": "auc", "subsample": 0.8, "colsample_bytree": 0.8,
    "min_child_weight": 20, "lambda": 1.0, "tree_method": "hist",
}


def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(MODELS.parent / "events" / f"{name}.parquet")


def feature_cols(df: pd.DataFrame | None = None) -> list[str]:
    cols = schema.load() or []
    return [c for c in cols if c in df.columns] if df is not None else cols


def fit_xgb(Xtr, ytr, Xva, yva, params: dict | None = None, rounds: int = 200):
    import xgboost as xgb
    p = dict(XGB_PARAMS)
    if params:
        p.update(params)
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)
    return xgb.train(p, dtr, num_boost_round=rounds, evals=[(dva, "valid")],
                     early_stopping_rounds=30, verbose_eval=False)


__all__ = ["load", "feature_cols", "fit_xgb", "auc", "XGB_PARAMS"]
