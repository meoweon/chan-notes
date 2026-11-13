"""E003：分数与收益的抵消机制，在样本外单独复核。

买卖点报告 §6.1 在全期（2021-01~2026-10）观察到：分数越高 → 存活率越高，
但「真点」的前向收益越低，两者抵消后分数对期望收益无梯度。该观察未做 IS/OOS 切分。
本脚本把同一口径拆成前段/后段各算一遍，检验抵消在样本外是否仍然成立。

口径完全沿用 analyze_bsppoint.py 的 load()：确认买点、有逐月 walk-forward 样本外分数、
2021-01~2026-10；前向收益用事件表已算好的 c_dir_ret_48（无组合交互）。
切分沿用 param_sweep_backtest.py 的 IS_END = 2025-07-01。

用法：
    EVENTS_NAME=crypto_31_v1 MODEL_NAME=wf_crypto_31_xgb_oos \
        .venv/bin/python scripts/analyze_score_return_oos.py

产物：
    reports/score_return_oos{OUT_SUFFIX}.json
"""
import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

NAME = os.environ.get("EVENTS_NAME", "crypto_31_v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "wf_crypto_31_xgb_oos")
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")

WINDOW_START = pd.Timestamp("2021-01-01")
IS_END = pd.Timestamp("2025-07-01")
OOS_END = pd.Timestamp("2026-11-01")
FW = "c_dir_ret_48"
B = 4000
SEED_CORR = 42
SEED_LIFT = 7
KEEP_RATIOS = (0.2, 0.4, 0.5, 0.6, 0.8)


def load():
    df = pd.read_parquet(DATA / "events" / f"{NAME}.parquet")
    with open(DATA / "models" / f"{MODEL_NAME}.pkl", "rb") as f:
        d = pickle.load(f)
    s = pd.Series(d["oos_prob"], index=d["oos_idx"])
    df["score"] = df.index.map(s)
    df["t"] = pd.to_datetime(df["event_time"])
    df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
    df = df[(df.t >= WINDOW_START) & (df.t < OOS_END) & df.score.notna()].copy()
    df["month"] = df.t.dt.to_period("M").astype(str)
    return df


def month_groups(months):
    return [np.where(months == m)[0] for m in np.unique(months)]


def deciles(g):
    q = pd.qcut(g["score"], 10, duplicates="drop", labels=False)
    rows = []
    for b in sorted(pd.Series(q).dropna().unique()):
        x = g[q == b]
        sr = float(x.survive_144.mean())
        t = float(x.loc[x.survive_144 == 1, FW].mean())
        f = float(x.loc[x.survive_144 == 0, FW].mean())
        rows.append({
            "bin": int(b), "n": int(len(x)),
            "score_lo": float(x.score.min()), "score_hi": float(x.score.max()),
            "survive_rate": sr, "fw_true": t, "fw_false": f,
            "fw_all": float(x[FW].mean()),
            "fw_rebuilt": sr * t + (1 - sr) * f,
            "n_true": int((x.survive_144 == 1).sum()),
        })
    bins = [r["bin"] for r in rows]
    rho = {k: float(spearmanr(bins, [r[k] for r in rows]).statistic)
           for k in ("survive_rate", "fw_true", "fw_all")}
    return rows, rho


def corr_boot(g, col, true_only=False):
    x = g[g.survive_144 == 1] if true_only else g
    sc = x.score.to_numpy()
    yy = x[col].to_numpy()
    groups = month_groups(x.month.to_numpy())
    rng = np.random.default_rng(SEED_CORR)
    iid, clu = [], []
    for _ in range(B):
        i = rng.integers(0, len(x), len(x))
        iid.append(spearmanr(sc[i], yy[i]).statistic)
        j = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        clu.append(spearmanr(sc[j], yy[j]).statistic)
    ci = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]
    return {
        "n": int(len(x)), "n_months": len(groups),
        "rho": float(spearmanr(sc, yy).statistic),
        "ci_iid": ci(iid), "ci_month": ci(clu),
    }


def lift_boot(g):
    sc = g.score.to_numpy()
    r = g[FW].to_numpy()
    groups = month_groups(g.month.to_numpy())
    rng = np.random.default_rng(SEED_LIFT)
    base = float(r.mean())
    out = []
    for keep in KEEP_RATIOS:
        thr = float(np.quantile(sc, 1 - keep))
        sel = sc >= thr
        draws = []
        for _ in range(B):
            j = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
            sj, rj = sc[j], r[j]
            draws.append(rj[sj >= np.quantile(sj, 1 - keep)].mean() - rj.mean())
        lo, hi = float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))
        out.append({
            "keep": keep, "thr": thr, "n_sel": int(sel.sum()),
            "fw_sel": float(r[sel].mean()), "lift": float(r[sel].mean() - base),
            "ci_month": [lo, hi], "ci_spans_zero": bool(lo < 0 < hi),
        })
    return base, out


def main():
    raw = load()
    out = {
        "events": NAME, "model": MODEL_NAME, "fw_col": FW,
        "window": [str(WINDOW_START.date()), str(OOS_END.date())],
        "is_end": str(IS_END.date()),
        "B": B, "seed_corr": SEED_CORR, "seed_lift": SEED_LIFT,
        "n_total": int(len(raw)), "windows": {},
    }
    for tag, sub in (("is", raw[raw.t < IS_END]), ("oos", raw[raw.t >= IS_END])):
        g = sub.dropna(subset=[FW, "score"])
        rows, rho = deciles(g)
        base, lifts = lift_boot(g)
        out["windows"][tag] = {
            "n": int(len(g)), "n_dropped_fw_nan": int(len(sub) - len(g)),
            "n_months": int(g.month.nunique()),
            "survive_rate": float(g.survive_144.mean()),
            "auc": float(roc_auc_score(g.survive_144, g.score)),
            "fw_all_mean": base,
            "deciles": rows, "spearman_decile": rho,
            "spearman_event": {
                "survive": corr_boot(g, "survive_144"),
                "fw_all": corr_boot(g, FW),
                "fw_true": corr_boot(g, FW, true_only=True),
            },
            "gating_lift": lifts,
        }
        print(f"[{tag}] n={len(g)} 月数={g.month.nunique()} 存活率={g.survive_144.mean():.3f} "
              f"AUC={out['windows'][tag]['auc']:.3f}")
        for k, v in out["windows"][tag]["spearman_event"].items():
            print(f"  {k:8s} rho={v['rho']:+.3f} 按月CI={v['ci_month']}")
        print(f"  门控全部跨0: {all(x['ci_spans_zero'] for x in lifts)}")

    dst = REPORTS / f"score_return_oos{OUT_SUFFIX}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"保存 {dst}")


if __name__ == "__main__":
    main()