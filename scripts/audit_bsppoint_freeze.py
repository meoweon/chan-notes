"""审计复算：买卖点报告口径 vs 蓝图 v7 冻结规格。

回答两个问题：
1. 报告的 48bar/2ATR/TP5ATR 换成冻结规格 144bar/3ATR/无止盈后，
   完美先知三分组（all / survive=1 / survive=0）的收益分离是否仍成立。
2. 可交易口径（score 月度事前门控 q60，非完美先知）在冻结规格下的 IS/OOS。

两种 IS/OOS 切分：
  - 报告口径（E001/E002）：IS 2021-01~2025-06，OOS 2025-07~2026-10
  - 蓝图口径：IS 2021-01~2023-12，OOS 2024-01~2026-10

用法：
    EVENTS_NAME=crypto_31_v1 MODEL_NAME=wf_crypto_31_xgb_oos \
        .venv/bin/python scripts/audit_bsppoint_freeze.py

产物：
    reports/bsppoint_freeze_audit.json
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import param_sweep_backtest as psb
from chan_notes.backtest import portfolio

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# 参数口径
P_REPORT = dict(hold=48, stop=2.0, tp=5.0)      # 报告口径（E001/E002 基线）
P_FROZEN = dict(hold=144, stop=3.0, tp=None)    # 蓝图 v7 冻结规格

# 切分窗口
W_REPORT = ("报告口径(2025-07切)", pd.Timestamp("2021-01-01"), pd.Timestamp("2025-07-01"), pd.Timestamp("2026-11-01"))
W_BLUE = ("蓝图口径(2024-01切)", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2026-11-01"))


def bt(ctx, mask, hold, stop, tp, s_name="无"):
    df = ctx["df"]
    if s_name != "无":
        month = df["t"].dt.to_period("M").astype(str)
        thr = month.map(ctx["s_thr"][s_name]).fillna(-np.inf)
        mask = mask & (df["score"].to_numpy() >= thr.to_numpy())
    sub = df[mask]
    if len(sub) < 10:
        return None
    p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=stop,
                           take_profit_atr=tp, hold_bars=hold, signal_exit=True)
    out = portfolio.run(sub, ctx["bars"], p, entry_mode="confirm")
    if out is None:
        return None
    return out["equity"], out["trades"], p


def cut(eq, tr, p, a, b):
    m = psb.slice_metrics(eq, tr, p, a, b)
    if m is None:
        return None
    return {k: m[k] for k in ("n", "ret", "sharpe", "mdd", "win", "avg_ret", "payoff")}


def fmt(m):
    if m is None:
        return "  (无)"
    return (f"  n={m['n']:>4}  ret={m['ret']*100:+8.1f}%  sharpe={m['sharpe']:+6.2f}  "
            f"win={m['win']*100:5.1f}%  payoff={m['payoff']:5.2f}  mdd={m['mdd']*100:6.1f}%")


def main():
    ctx = psb.build_context(verbose=True)
    df = ctx["df"]
    masks = {
        "all": np.ones(len(df), bool),
        "true(survive=1)": (df["survive_144"] == 1).to_numpy(),
        "false(survive=0)": (df["survive_144"] == 0).to_numpy(),
    }

    out = {"n_events": int(len(df)), "survive_rate": float(df["survive_144"].mean()),
           "params": {"report": P_REPORT, "frozen": P_FROZEN}, "results": {}}

    combos = []
    for gname, mask in masks.items():
        combos.append((f"{gname} @ 报告48/2.0/TP5", mask, P_REPORT, "无"))
        combos.append((f"{gname} @ 冻结144/3.0/无TP", mask, P_FROZEN, "无"))
        # 止盈 on/off 干净配对：同 hold/stop，只差 take_profit
        combos.append((f"{gname} @ 48/2.0/无TP", mask, dict(hold=48, stop=2.0, tp=None), "无"))
        combos.append((f"{gname} @ 冻结144/3.0/TP5", mask, dict(hold=144, stop=3.0, tp=5.0), "无"))
    # 可交易口径：score 月度事前门控 q60（非完美先知）
    combos.append(("score>=月度q60 @ 冻结144/3.0/无TP", np.ones(len(df), bool), P_FROZEN, "q60"))
    combos.append(("score>=月度q60 @ 报告48/2.0/TP5", np.ones(len(df), bool), P_REPORT, "q60"))

    for name, mask, par, s_name in combos:
        r = bt(ctx, mask, par["hold"], par["stop"], par["tp"], s_name)
        rec = {}
        print(f"\n== {name}" + (f"（score 门控 {s_name}）" if s_name != "无" else "") + " ==")
        if r is None:
            print("  (回测无结果)")
            out["results"][name] = None
            continue
        eq, tr, p = r
        for wname, t0, t1, t2 in (W_REPORT, W_BLUE):
            is_m = cut(eq, tr, p, t0, t1)
            oos_m = cut(eq, tr, p, t1, t2)
            rec[wname] = {"is": is_m, "oos": oos_m}
            print(f" [{wname}] IS : {fmt(is_m)}")
            print(f" [{wname}] OOS: {fmt(oos_m)}")
        out["results"][name] = rec

    dst = REPORTS / "bsppoint_freeze_audit.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    print(f"\n保存 {dst}")


if __name__ == "__main__":
    main()
