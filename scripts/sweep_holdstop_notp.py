"""E004：hold×stop 网格样本外扫描（无止盈）。

预注册：docs/30_实验/E004-holdstop网格样本外.md
回答：基线 48bar/2ATR 在 36 格里处于什么位置；IS 选出的最优格能否在 OOS 复现；
是否存在参数不敏感的稳健平台。

用法：
    EVENTS_NAME=crypto_31_v1 MODEL_NAME=wf_crypto_31_xgb_oos \
        .venv/bin/python scripts/sweep_holdstop_notp.py

产物：
    reports/holdstop_sweep_notp.json
    reports/holdstop_sweep_notp.csv
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

HOLDS = [24, 48, 96, 144, 192, 288]          # 1h bar：1/2/4/6/8/12 天
STOPS = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]       # ×ATR
TP = None                                     # 台账 #47：止盈禁用

WINDOWS = {
    "cut2025-07": (pd.Timestamp("2021-01-01"), pd.Timestamp("2025-07-01"), pd.Timestamp("2026-11-01")),
    "cut2024-01": (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2026-11-01")),
}
KEYS = ("n", "ret", "sharpe", "mdd", "win", "avg_ret", "payoff")


def bt(ctx, hold, stop):
    df = ctx["df"]
    p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=stop,
                           take_profit_atr=TP, hold_bars=hold, signal_exit=True)
    out = portfolio.run(df, ctx["bars"], p, entry_mode="confirm")
    if out is None:
        return None
    return out["equity"], out["trades"], p


def cut(eq, tr, p, a, b):
    m = psb.slice_metrics(eq, tr, p, a, b)
    if m is None:
        return None
    return {k: m[k] for k in KEYS}


def main():
    ctx = psb.build_context(verbose=True)
    print(f"事件数 {len(ctx['df'])}，网格 {len(HOLDS)}×{len(STOPS)}={len(HOLDS)*len(STOPS)} 格，tp=None")

    rows, grid = [], {}
    for hold in HOLDS:
        for stop in STOPS:
            tag = f"h{hold}/s{stop}"
            r = bt(ctx, hold, stop)
            if r is None:
                print(f"{tag}: 回测无结果")
                continue
            eq, tr, p = r
            rec = {}
            row = {"hold": hold, "stop": stop}
            for wname, (t0, t1, t2) in WINDOWS.items():
                is_m = cut(eq, tr, p, t0, t1)
                oos_m = cut(eq, tr, p, t1, t2)
                rec[wname] = {"is": is_m, "oos": oos_m}
                for seg, m in (("is", is_m), ("oos", oos_m)):
                    for k in KEYS:
                        row[f"{wname}_{seg}_{k}"] = None if m is None else m[k]
                def f(m):
                    return "(无)" if m is None else (
                        f"n={m['n']:>4} ret={m['ret']*100:+8.1f}% sharpe={m['sharpe']:+6.2f} "
                        f"win={m['win']*100:5.1f}% mdd={m['mdd']*100:6.1f}%")
                print(f"{tag} [{wname}] IS {f(is_m)} | OOS {f(oos_m)}")
            grid[tag] = rec
            rows.append(row)

    out = {"n_events": int(len(ctx["df"])), "tp": TP,
           "holds": HOLDS, "stops": STOPS,
           "windows": {k: [str(x) for x in v] for k, v in WINDOWS.items()},
           "grid": grid}
    dst = REPORTS / "holdstop_sweep_notp.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str))
    pd.DataFrame(rows).to_csv(REPORTS / "holdstop_sweep_notp.csv", index=False)
    print(f"\n保存 {dst} 与 holdstop_sweep_notp.csv（{len(rows)} 行）")


if __name__ == "__main__":
    main()
