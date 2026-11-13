"""特征体检与筛选。

四项证据：
1. 单变量 AUC（含 90% bootstrap 区间）—— 对"盈利"和"大波(>2%)"两个标签
2. 年度方向稳定性 —— 逐年 AUC 是否同向
3. 多变量贡献 —— XGBoost gain 与 permutation importance
4. 冗余聚类 —— |Spearman|≥0.8 的列只保留信息更强的一支
5. 分组消融 —— 去掉整组后大波 AUC 的下降
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import xgboost as xgb

from chan_notes.eval.metrics import auc, auc_ci
from chan_notes.features import schema
from chan_notes.models import train

SPLIT = "2023-06-30"
GROUPS = {
    "pa": "价格行为", "pt": "形态", "ch": "缠论", "mkt": "市场",
    "bc": "bar计数", "sw": "摆动/环境", "rg": "环境标签", "bsw": "突破强度",
    "ff": "旗形", "rf": "区间边缘", "wg": "楔形",
}
BROOKS2 = ("bc", "sw", "rg", "bsw", "ff", "rf", "wg")
OUT_CSV = str(_Path(__file__).resolve().parents[1] / "reports" / "feature_selection.csv")


def load_conf_buy(name: str):
    df = train.load(name)
    df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
    df["t"] = pd.to_datetime(df["event_time"])
    cols = train.feature_cols(df)
    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df, cols, X


def univariate(df, cols, X):
    y_pos = (df["c_pol_dir_ret"] > 0).astype(int).to_numpy()
    y_big = (df["c_pol_dir_ret"] > 0.02).astype(int).to_numpy()
    year = df["t"].dt.year
    rows = []
    for c in cols:
        s = X[c].to_numpy()
        a_pos, a_big = auc(y_pos, s), auc(y_big, s)
        lo, hi = auc_ci(y_pos, s, n_boot=80)
        by = [auc(y_pos[(year == yr).to_numpy()], s[(year == yr).to_numpy()])
              for yr in sorted(year.unique()) if (year == yr).sum() >= 80]
        by = [v for v in by if v == v]
        stable = sum(1 for v in by if (v - 0.5) * (a_pos - 0.5) > 0) / len(by) if len(by) >= 3 else np.nan
        rows.append({"feature": c, "group": c.split("_")[0], "auc_pos": a_pos,
                     "ci_lo": lo, "ci_hi": hi, "auc_big": a_big,
                     "sign_stable": stable, "std": float(s.std())})
    return pd.DataFrame(rows)


def multivariate(df, cols, X):
    tr = df["t"] <= pd.Timestamp(SPLIT)
    te = ~tr
    keep = [c for c in cols if X.loc[tr, c].nunique() > 1]
    y_pos = (df["c_pol_dir_ret"] > 0).astype(int).to_numpy()
    y_big = (df["c_pol_dir_ret"] > 0.02).astype(int).to_numpy()

    bst = train.fit_xgb(X.loc[tr, keep], y_pos[tr], X.loc[te, keep], y_pos[te], rounds=150)
    auc_pos = auc(y_pos[te], bst.predict(xgb.DMatrix(X.loc[te, keep])))
    gain = {keep[int(k[1:])]: v for k, v in bst.get_score(importance_type="gain").items() if k.startswith("f")}
    total = sum(gain.values()) or 1.0

    bst_big = train.fit_xgb(X.loc[tr, keep], y_big[tr], X.loc[te, keep], y_big[te], rounds=150)
    auc_big = auc(y_big[te], bst_big.predict(xgb.DMatrix(X.loc[te, keep])))
    gain_big = {keep[int(k[1:])]: v for k, v in bst_big.get_score(importance_type="gain").items() if k.startswith("f")}
    total_big = sum(gain_big.values()) or 1.0

    rng = np.random.default_rng(1)
    perm = {}
    base = bst.predict(xgb.DMatrix(X.loc[te, keep]))
    for c in keep:
        Xp = X.loc[te, keep].copy()
        Xp[c] = rng.permutation(Xp[c].to_numpy())
        perm[c] = auc(y_pos[te], base) - auc(y_pos[te], bst.predict(xgb.DMatrix(Xp)))

    abl = [{"removed": "none", "auc_big": auc_big}]
    for g in GROUPS:
        sub = [c for c in keep if not c.startswith(g + "_")]
        if len(sub) == len(keep):
            continue
        b = train.fit_xgb(X.loc[tr, sub], y_big[tr], X.loc[te, sub], y_big[te], rounds=150)
        abl.append({"removed": g, "auc_big": auc(y_big[te], b.predict(xgb.DMatrix(X.loc[te, sub])))})
    sub2 = [c for c in keep if c.split("_")[0] not in BROOKS2]
    b2 = train.fit_xgb(X.loc[tr, sub2], y_big[tr], X.loc[te, sub2], y_big[te], rounds=150)
    abl.append({"removed": "brooks2_all",
                "auc_big": auc(y_big[te], b2.predict(xgb.DMatrix(X.loc[te, sub2])))})
    return {"auc_pos": auc_pos, "auc_big": auc_big, "keep": keep, "gain": gain, "total": total,
            "gain_big": gain_big, "total_big": total_big, "perm": perm, "ablation": pd.DataFrame(abl),
            "auc_big_old_only": auc(y_big[te], b2.predict(xgb.DMatrix(X.loc[te, sub2]))),
            "n_old_only": len(sub2)}


def cluster(out, X, keep):
    corr = X[keep].corr(method="spearman").abs()
    out["redundant_with"] = ""
    kept = []
    for c in out.sort_values("auc_big", key=lambda s: (s - 0.5).abs(), ascending=False)["feature"]:
        if c not in keep:
            continue
        dup = next((k for k in kept if corr.loc[c, k] >= 0.8), None)
        if dup:
            out.loc[out.feature == c, "redundant_with"] = dup
        else:
            kept.append(c)
    return out


def main(name="events_v2"):
    df, cols, X = load_conf_buy(name)
    uni = univariate(df, cols, X)
    mv = multivariate(df, cols, X)

    uni["perm_imp"] = uni["feature"].map(mv["perm"])
    uni["gain_pos"] = uni["feature"].map(mv["gain"]).fillna(0) / mv["total"]
    uni["gain_big"] = uni["feature"].map(mv["gain_big"]).fillna(0) / mv["total_big"]
    out = cluster(uni, X, mv["keep"])
    out["kept_after_cluster"] = ~out["redundant_with"].astype(bool)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print(f"样本={len(df)}  特征={len(cols)}  test_AUC 正收益={mv['auc_pos']:.4f}  大波={mv['auc_big']:.4f}")
    print(f"旧特征集({mv['n_old_only']} 列) 大波 AUC={mv['auc_big_old_only']:.4f}  "
          f"新增 brooks 增量={mv['auc_big'] - mv['auc_big_old_only']:+.4f}")
    print("\n## 分组消融（大波 AUC）")
    print(mv["ablation"].round(4).to_string(index=False))

    eff = out[(out.perm_imp > 0.002) & (out.sign_stable >= 0.6)].sort_values("perm_imp", ascending=False)
    print("\n## 多变量有效（perm_imp>0.002 且年度方向一致≥60%）")
    print(eff[["feature", "auc_pos", "auc_big", "sign_stable", "perm_imp", "gain_big"]].head(30).round(3).to_string(index=False))

    sig = out[(out.ci_lo > 0.5) | (out.ci_hi < 0.5)].sort_values("auc_pos")
    print(f"\n## 单变量显著（90%CI 不含 0.5）：{len(sig)} 个")
    print(sig[["feature", "auc_pos", "ci_lo", "ci_hi", "sign_stable", "auc_big"]].round(3).to_string(index=False))

    print(f"\n聚类冗余：保留 {int(out.kept_after_cluster.sum())}/{len(cols)}，"
          f"高冗余 {int((~out.kept_after_cluster).sum())} 列")
    dead = out[(out.std == 0) | (out.perm_imp.abs() < 1e-6) & (out.gain_pos == 0) & (out.gain_big == 0)]
    print(f"零贡献列：{len(dead)}  -> {', '.join(dead.feature.tolist()[:20])}")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["events_v2"]))
