"""买卖点分析：分类型前向收益、分数阈值扫描、144/多 horizon 合理性。

只读 data/events/{EVENTS_NAME}.parquet + data/models/{MODEL_NAME}.pkl（逐月 walk-forward
样本外预测）。口径与 validate_bsppoint_profit.py 一致：确认买点、有样本外分数、
2021-01 ~ 2026-11。所有前向收益用事件表里已算好的列，不再做组合回测。

用法：
    EVENTS_NAME=crypto_21_v1 MODEL_NAME=wf_crypto_21_xgb_oos \
        .venv/bin/python scripts/analyze_bsppoint.py

产物：
    reports/bsppoint_analysis{OUT_SUFFIX}.json
"""
import os
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
NAME = os.environ.get("EVENTS_NAME", "crypto_21_v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "wf_crypto_21_xgb_oos")
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")

WINDOW_START = pd.Timestamp("2021-01-01")
OOS_END = pd.Timestamp("2026-11-01")

# 前向收益口径：确认入场、按方向取号、扣 13bp。48bar 为基线持有期。
FW = "c_dir_ret_48"          # 持有 48bar 收盘离场（无止损退出，只有命中标志）
POL = "c_pol_dir_ret"        # 策略退出（2ATR 止损 or 24bar 时间）
HORIZONS = [8, 24, 48, 96, 144, 288, 576]


def load():
    df = pd.read_parquet(DATA / "events" / f"{NAME}.parquet")
    with open(DATA / "models" / f"{MODEL_NAME}.pkl", "rb") as f:
        d = pickle.load(f)
    s = pd.Series(d["oos_prob"], index=d["oos_idx"])
    df["score"] = df.index.map(s)
    df["t"] = pd.to_datetime(df["event_time"])
    df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
    df = df[(df.t >= WINDOW_START) & (df.t < OOS_END) & df.score.notna()].copy()
    return df


def survive_h(df, h):
    """144 口径推广到任意 horizon：clear_bar 为空(-1)或 > bar_idx+h 即视为存活到 h。"""
    cb = df["clear_bar"].to_numpy()
    bi = df["bar_idx"].to_numpy()
    return (cb == -1) | (cb > bi + h)


def rstat(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return None
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "win": float((x > 0).mean()),
    }


def main():
    df = load()
    print(f"分析子集 {len(df)}（确认买点、有 OOS 分数、2021-01~2026-10）")
    out = {
        "n": int(len(df)),
        "survive_rate_144": float(df["survive_144"].mean()),
        "auc_144": float(roc_auc_score(df["survive_144"], df["score"])),
        "types": {},
        "deciles": [],
        "ratios": [],
        "horizons": [],
        "lifespan": {},
    }

    # ---- 分类型 ----
    for tp, g in df.groupby("bsp_type"):
        auc = None
        if len(g) >= 30 and g["survive_144"].nunique() > 1:
            auc = float(roc_auc_score(g["survive_144"], g["score"]))
        t = g[g.survive_144 == 1]
        f = g[g.survive_144 == 0]
        sel = g["score"] >= 0.5
        tp_, fp_ = int((sel & (g.survive_144 == 1)).sum()), int((sel & (g.survive_144 == 0)).sum())
        fn_, tn_ = int((~sel & (g.survive_144 == 1)).sum()), int((~sel & (g.survive_144 == 0)).sum())
        prec = tp_ / max(tp_ + fp_, 1)
        rec = tp_ / max(tp_ + fn_, 1)
        out["types"][tp] = {
            "n": int(len(g)),
            "survive_rate": float(g["survive_144"].mean()),
            "auc": auc,
            "fw_all": rstat(g[FW]),
            "fw_true": rstat(t[FW]),
            "fw_false": rstat(f[FW]),
            "pol_all": rstat(g[POL]),
            "thr": 0.5, "tp": tp_, "fp": fp_, "fn": fn_, "tn": tn_,
            "prec": prec, "rec": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9),
        }

    # ---- 分数十分位：存活率 + 前向收益 ----
    try:
        qbins = pd.qcut(df["score"], 10, duplicates="drop", labels=False)
    except Exception:
        qbins = pd.Series(np.nan, index=df.index)
    for b in sorted(pd.Series(qbins).dropna().unique()):
        g = df[qbins == b]
        out["deciles"].append({
            "bin": int(b),
            "score_lo": float(g["score"].min()),
            "score_hi": float(g["score"].max()),
            "n": int(len(g)),
            "survive_rate": float(g["survive_144"].mean()),
            "fw": rstat(g[FW]),
            "fw_true": rstat(g[g.survive_144 == 1][FW]),
            "fw_false": rstat(g[g.survive_144 == 0][FW]),
        })

    # ---- 阈值 + 四个比例（阳性率 / recall / FPR / TP占比）----
    y = df["survive_144"].to_numpy()
    s = df["score"].to_numpy()
    thrs = [("0.5", 0.5)] + [(f"q{int(q*100)}", float(np.quantile(s, q))) for q in (0.4, 0.6, 0.8, 0.9)]
    # 约登指数最优阈值（网格 0.3~0.8）
    best_j, best_jt = -1, None
    grid = np.linspace(0.3, 0.8, 51)
    for t in grid:
        pred = s >= t
        tp = ((pred) & (y == 1)).sum(); fp = ((pred) & (y == 0)).sum()
        fn = ((~pred) & (y == 1)).sum(); tn = ((~pred) & (y == 0)).sum()
        j = tp / max(tp + fn, 1) - fp / max(fp + tn, 1)
        if j > best_j:
            best_j, best_jt = j, float(t)
    thrs.append((f"约登最优≈{best_jt:.3f}", best_jt))
    for name, t in thrs:
        pred = s >= t
        tp = int(((pred) & (y == 1)).sum()); fp = int(((pred) & (y == 0)).sum())
        fn = int(((~pred) & (y == 1)).sum()); tn = int(((~pred) & (y == 0)).sum())
        n = tp + fp + fn + tn
        recall = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * recall / max(prec + recall, 1e-9)
        out["ratios"].append({
            "name": name, "thr": t, "n_pos": int(tp + fp),
            "阳性率": float((tp + fp) / n), "recall": float(recall),
            "FPR": float(fpr), "precision": float(prec),
            "TP占比": float(tp / n), "f1": float(f1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })

    # ---- 多 horizon：存活率 + 模型 AUC ----
    for h in HORIZONS:
        yh = survive_h(df, h)
        a = float(roc_auc_score(yh, df["score"])) if len(np.unique(yh)) > 1 else None
        out["horizons"].append({
            "h": h, "survive_rate": float(yh.mean()), "auc": a,
        })

    # ---- 分类型中位寿命（被擦除事件的 lifespan_bars；存活事件 = NaN）----
    for tp, g in df.groupby("bsp_type"):
        ls = g.loc[g["cleared_ever"] == 1, "lifespan_bars"].dropna()
        out["lifespan"][tp] = {
            "n_cleared": int(len(ls)),
            "median": float(ls.median()) if len(ls) else None,
            "p75": float(ls.quantile(0.75)) if len(ls) else None,
        }

    # ---- 四根 bar：锚点/信号重合 + 确认等待（分类型）----
    raw = pd.read_parquet(DATA / "events" / f"{NAME}.parquet")
    raw["t"] = pd.to_datetime(raw["event_time"])
    raw = raw[(raw.is_buy == 1) & (raw.t >= WINDOW_START) & (raw.t < OOS_END)].copy()
    out["bars"] = {}
    for tp in list(df.bsp_type.value_counts().index) + ["全部"]:
        g = df if tp == "全部" else df[df.bsp_type == tp]
        rg = raw if tp == "全部" else raw[raw.bsp_type == tp]
        w = g.conf_wait.value_counts().sort_index()
        pos = g[g.delay_bars > 0]
        out["bars"][tp] = {
            "n": int(len(g)),
            "n_buy": int(len(rg)),
            "conf_fail": int((rg.conf_ok == 0).sum()),
            "delay0": int((g.delay_bars == 0).sum()),
            "delay_pos": int(len(pos)),
            "delay_max": int(g.delay_bars.max()) if len(g) else 0,
            "delay_med_pos": float(pos.delay_bars.median()) if len(pos) else None,
            "wait": {str(int(k)): int(v) for k, v in w.items()},
            "wait_med": float(g.conf_wait.median()) if len(g) else None,
            "wait_mean": float(g.conf_wait.mean()) if len(g) else None,
        }

    dst = REPORTS / f"bsppoint_analysis{OUT_SUFFIX}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"保存 {dst}")
    # 简短自检
    print("horizons:", [(h["h"], round(h["survive_rate"], 3), round(h["auc"], 3)) for h in out["horizons"]])


if __name__ == "__main__":
    main()
