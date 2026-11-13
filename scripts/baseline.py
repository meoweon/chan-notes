"""事件研究基线：总体 / 类型 / 年份 / 退出网格 / 确认入场。

用法:
  python baseline.py --name events_v2 [--buy-only] [--grid 8,24,48]
"""
import argparse
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import itertools
import numpy as np
import pandas as pd

from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many
from chan_notes.eval.metrics import summarize
from chan_notes.models import train


def _rets(df, arrays, hold, stop, cost=0.0013):
    out = []
    for r in df.itertuples():
        b = arrays[r.code]
        i0 = int(r.entry_idx)
        res = simulate_exit(b, i0, int(r.is_buy), float(r.atr), hold=hold, stop_atr=stop, cost=cost)
        if res is not None:
            out.append(res.ret if r.is_buy else -res.ret)
    return pd.Series(out)


def _stat(s: pd.Series) -> dict:
    se = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
    return {"n": len(s), "mean_bp": s.mean() * 1e4, "median_bp": s.median() * 1e4,
            "win": (s > 0).mean(), "t": s.mean() / se if se and se > 0 else np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="events_v2")
    ap.add_argument("--freq", default="30m")
    ap.add_argument("--grid", default="8,24,48")
    args = ap.parse_args()

    df = train.load(args.name)
    arrays = {c: for_policy(a) for c, a in load_many(df.code.unique(), args.freq).items()}
    holds = [int(x) for x in args.grid.split(",")]

    print(f"# {args.name}  n={len(df)}  {df.event_time.min()} ~ {df.event_time.max()}")
    print(f"买点 {int(df.is_buy.sum())}  卖点 {int((1 - df.is_buy).sum())}")

    print("\n## 按类型（信号次 bar 开盘入场，持 24 bar，2×ATR 止损）")
    df = df.copy()
    df["base_ret"] = _rets(df, arrays, 24, 2.0)
    tab = []
    for side, sd in ((1, "BUY"), (0, "SELL")):
        sub = df[df.is_buy == side]
        tab.append({"group": sd, **_stat(sub["base_ret"])})
        for t in ("1", "1p", "2", "2s", "3a", "3b"):
            s2 = sub[sub.bsp_type.str.split(",").apply(lambda xs: t in xs)]
            if len(s2) >= 50:
                tab.append({"group": f"{sd}_{t}", **_stat(s2["base_ret"])})
    print(pd.DataFrame(tab).round(1).to_string(index=False))

    print("\n## 退出网格（dir_ret, bp）")
    rows = []
    for hold, stop in itertools.product(holds, (1.0, 1.5, 2.0, 3.0)):
        rows.append({"hold": hold, "stop_atr": stop, **_stat(_rets(df, arrays, hold, stop))})
    grid = pd.DataFrame(rows)
    print(grid.round(1).to_string(index=False))
    buy = df[df.is_buy == 1]
    rows = []
    for hold, stop in itertools.product(holds, (1.0, 1.5, 2.0, 3.0)):
        rows.append({"hold": hold, "stop_atr": stop, **_stat(_rets(buy, arrays, hold, stop))})
    print("\n## 退出网格（仅买点）")
    print(pd.DataFrame(rows).round(1).to_string(index=False))

    print("\n## 确认入场（仅买点；信号后 N bar 内突破信号 bar 高点，次 bar 开盘入场）")
    rows = []
    for wait, hold, stop in itertools.product((1, 2, 4), holds, (1.5, 2.0)):
        sub = buy[(buy.conf_ok == 1) & (buy.conf_wait <= wait)]
        rets = _rets_confirmed(sub, arrays, hold, stop)
        rows.append({"wait": wait, "hold": hold, "stop": stop, **_stat(rets)})
    print(pd.DataFrame(rows).round(1).to_string(index=False))


def _rets_confirmed(df, arrays, hold, stop, cost=0.0013):
    out = []
    for r in df.itertuples():
        b = arrays[r.code]
        res = simulate_exit(b, int(r.conf_entry_idx), 1, float(r.atr), hold=hold, stop_atr=stop, cost=cost)
        if res is not None:
            out.append(res.ret)
    return pd.Series(out)


if __name__ == "__main__":
    main()
