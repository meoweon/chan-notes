"""分数 vs 前向波动率：检验"高分 = 低波动(没肉)、低分 = 高波动(有肉)"。

前向波动率的可用代理（事件表已算好）：`c_mfe_48 - c_mae_48`，即确认入场后 48bar
窗口内 (最高 - 最低)/入场价，是"这段行情摆了多大"（振幅/range，非标准差）。
同时拆成上行 `c_mfe_48` 与下行 `|c_mae_48|`，看波动是涨出来的还是跌出来的。

只读 data/events/{EVENTS_NAME}.parquet + data/models/{MODEL_NAME}.pkl，口径与
analyze_bsppoint.py 一致：确认买点、有 OOS 分数、2021-01~2026-11、删失行(c_mfe_48 为 NaN)剔除。

用法：
    EVENTS_NAME=crypto_21_v1 MODEL_NAME=wf_crypto_21_xgb_oos \
        .venv/bin/python scripts/analyze_score_vol.py
产物：
    reports/bsppoint_score_vol{OUT_SUFFIX}.json
"""
import os
import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

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


def load():
    df = pd.read_parquet(DATA / "events" / f"{NAME}.parquet")
    with open(DATA / "models" / f"{MODEL_NAME}.pkl", "rb") as f:
        d = pickle.load(f)
    s = pd.Series(d["oos_prob"], index=d["oos_idx"])
    df["score"] = df.index.map(s)
    df["t"] = pd.to_datetime(df["event_time"])
    df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
    df = df[(df.t >= WINDOW_START) & (df.t < OOS_END) & df.score.notna()].copy()
    # 前向波动率需要完整 48bar 窗口（末端删失 → NaN）
    df["amp"] = df["c_mfe_48"] - df["c_mae_48"]      # 48bar 振幅（range）
    df["up"] = df["c_mfe_48"]                         # 上行（买=向上最大幅度）
    df["dn"] = -df["c_mae_48"]                        # 下行（买=向下最大幅度，取正）
    df = df[df["amp"].notna()].copy()
    return df


def spearman(a, b):
    a = pd.Series(a); b = pd.Series(b)
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    return float(a.rank().corr(b.rank()))


def main():
    df = load()
    n = len(df)
    surv = df["survive_144"].astype(bool)
    tru = df[surv]
    fls = df[~surv]

    def stats(g):
        return {
            "n": int(len(g)),
            "amp_mean": float(g["amp"].mean()),
            "amp_med": float(g["amp"].median()),
            "up_mean": float(g["up"].mean()),
            "dn_mean": float(g["dn"].mean()),
        }

    tru = tru.copy()
    tru["up_share"] = tru["up"] / tru["amp"]

    def type_corr(col):
        r = {}
        for tp, g in tru.groupby("bsp_type"):
            if len(g) >= 30:
                r[tp] = float(spearman(g["score"], g[col]))
        return r

    type_true_n = {tp: int(len(g)) for tp, g in tru.groupby("bsp_type") if len(g) >= 30}

    out = {
        "n": n,
        "metric": "amp = c_mfe_48 - c_mae_48（确认入场后 48bar 的 (high-low)/entry 振幅，%）；up_share = c_mfe_48/amp（上行占比）",
        "all": stats(df),
        "true": stats(tru),
        "false": stats(fls),
        "spearman_score_vs_amp_all": spearman(df["score"], df["amp"]),
        "spearman_score_vs_amp_true": spearman(tru["score"], tru["amp"]),
        "spearman_score_vs_amp_false": spearman(fls["score"], fls["amp"]),
        "spearman_score_vs_up_true": spearman(tru["score"], tru["up"]),
        "spearman_score_vs_dn_true": spearman(tru["score"], tru["dn"]),
        "spearman_score_vs_up_share_true": spearman(tru["score"], tru["up_share"]),
        "spearman_score_vs_fw_true": spearman(tru["score"], tru["c_dir_ret_48"]),
        "type_true_n": type_true_n,
        "type_corr_dn": type_corr("dn"),
        "type_corr_up_share": type_corr("up_share"),
        "type_corr_fw": type_corr("c_dir_ret_48"),
        "deciles": [],
    }

    q = pd.qcut(df["score"], 10, labels=False, duplicates="drop")
    df["decile"] = q
    for b in sorted(df["decile"].unique()):
        g = df[df.decile == b]
        gt = g[g.survive_144 == 1]
        gf = g[g.survive_144 == 0]
        out["deciles"].append({
            "bin": int(b),
            "score_lo": float(g["score"].min()),
            "score_hi": float(g["score"].max()),
            "n": int(len(g)),
            "survive_rate": float(g["survive_144"].mean()),
            "amp_all": float(g["amp"].mean()),
            "amp_true": float(gt["amp"].mean()) if len(gt) else None,
            "up_true": float(gt["up"].mean()) if len(gt) else None,
            "dn_true": float(gt["dn"].mean()) if len(gt) else None,
            "fw_true": float(gt["c_dir_ret_48"].mean()) if len(gt) else None,
        })

    dst = REPORTS / f"bsppoint_score_vol{OUT_SUFFIX}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"保存 {dst}  n={n}")
    print(f"整体：amp 均值 {out['all']['amp_mean']*100:.2f}%  上行 {out['all']['up_mean']*100:.2f}%  下行 {out['all']['dn_mean']*100:.2f}%")
    print(f"真(存活)：amp {out['true']['amp_mean']*100:.2f}%  |  假(擦除)：amp {out['false']['amp_mean']*100:.2f}%")
    print(f"Spearman(score, amp)：全部 {out['spearman_score_vs_amp_all']:+.3f}  真 {out['spearman_score_vs_amp_true']:+.3f}  假 {out['spearman_score_vs_amp_false']:+.3f}")
    print(f"Spearman(score, 上行|真) {out['spearman_score_vs_up_true']:+.3f}   (score, 下行|真) {out['spearman_score_vs_dn_true']:+.3f}")
    print("\n  分位  存活率  amp全    amp真   上行真  下行真  真前向收益")
    for r in out["deciles"]:
        def p(x): return "—" if x is None else f"{x*100:5.2f}%"
        print(f"  D{r['bin']:>1}  {r['survive_rate']*100:4.0f}%  {p(r['amp_all'])}  {p(r['amp_true'])}  {p(r['up_true'])}  {p(r['dn_true'])}  {p(r['fw_true'])}")


if __name__ == "__main__":
    main()
