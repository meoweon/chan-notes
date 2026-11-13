"""可复现：买卖点「真(存活)→赚钱」假设验证。

核心问题：缠论买卖点在被发现的当下，无法知道它之后会不会被擦除（重绘）。
本脚本用「完美先知」——未来标签 survive_144（144 bar 内是否被擦除）——
把确认买点事件分成 真 / 假 两组，用同一套基线参数分别回测，
检验「存活(真)的点赚钱、被擦除(假)的点亏钱」是否在样本内和样本外都成立。

这是假设验证，不是策略：survive_144 是 144 bar 之后才知道的标签，
结论只能说明「标签有经济含义」，不能说明「当前可交易」。

用法：
    EVENTS_NAME=crypto_21_v1 MODEL_NAME=wf_crypto_21_xgb_oos \
        .venv/bin/python scripts/validate_bsppoint_profit.py

产物：
    reports/bsppoint_profit_validation{OUT_SUFFIX}.json
"""
import os
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import param_sweep_backtest as psb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")

# 基线参数：hold/stop 沿用 E001/E002 基线 48/2.0。
# 止盈已禁用（None）：2026-11-09 止盈 on/off 干净对照（台账 #47、
# reports/bsppoint_audit_20261109.md §3）——48/2.0 下同 hold/stop 配对，
# TP5 在 IS 与 OOS、两种切分上一致差于无止盈
# （OOS 夏普 −0.05 vs +0.41（2025-07 切）/ −0.01 vs +0.47（2024-01 切））；
# 收益分布为长尾（台账 #16/#20），固定止盈=系统性截断右尾。
HOLD, STOP, TP = 48, 2.0, None


def fmt_split(rec, tag):
    s = rec[tag]
    return (f"  {tag.upper():4} 笔={s['n']:>4}  收益={s['ret']*100:+7.1f}%  "
            f"夏普={s['sharpe']:+5.2f}  每笔={s['avgret']*100:+6.2f}%  "
            f"胜率={s['win']*100:5.1f}%  盈亏比={s['payoff']:5.2f}  MDD={s['mdd']*100:6.1f}%")


def main():
    ctx = psb.build_context(verbose=True)
    df = ctx["df"].copy()

    auc = roc_auc_score(df["survive_144"], df["score"])
    out = {
        "n_events": int(len(df)),
        "survive_rate": float(df["survive_144"].mean()),
        "model_auc_vs_survive": float(auc),
        "params": {"hold": HOLD, "stop": STOP, "take_profit": TP},
        "splits": {},
        "types": {},
    }
    print(f"事件 {out['n_events']}  存活率 {out['survive_rate']:.4f}  "
          f"模型 AUC vs survive_144 = {auc:.4f}")

    # 1) 真/假 分组回测（完美先知）
    splits = [("all", np.ones(len(df), bool)),
              ("true", (df["survive_144"] == 1).values),
              ("false", (df["survive_144"] == 0).values)]
    for name, mask in splits:
        c = dict(ctx)
        c["df"] = df[mask]
        r = psb.run_combo(c, HOLD, STOP, TP, "无", "无")
        if r is None:
            out["splits"][name] = None
            print(f"\n== {name} == (无结果)")
            continue
        rec = {"n_all": r["n_all"], "ret_all": r["ret_all"], "sharpe_all": r["sharpe_all"]}
        for tag in ("is", "oos"):
            rec[tag] = {k: r.get(f"{k}_{tag}") for k in
                        ("n", "ret", "sharpe", "mdd", "win", "avgret", "payoff", "freq")}
        out["splits"][name] = rec
        print(f"\n== {name} ==")
        for tag in ("is", "oos"):
            print(fmt_split(rec, tag))

    # 2) 分类型计数 / 存活率 / 模型 AUC（报告骨架用；口径 = 确认买点、有 OOS 分数的子集）
    for tp, g in df.groupby("bsp_type"):
        a = None
        if len(g) >= 30 and g["survive_144"].nunique() > 1:
            a = float(roc_auc_score(g["survive_144"], g["score"]))
        out["types"][tp] = {
            "n": int(len(g)),
            "survive_rate": float(g["survive_144"].mean()),
            "buy_ratio": float(g["is_buy"].mean()),
            "auc": a,
        }

    dst = REPORTS / f"bsppoint_profit_validation{OUT_SUFFIX}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"\n保存 {dst}")


if __name__ == "__main__":
    main()
