"""卖点过滤器实验：买点前 N 根内出现过卖点，就不开仓。

依据：卖点后 24 bar 平均下跌 14.8bp（t=5.7），是当前唯一稳定的方向性信息；
A股不能做空，但可以把它当作多头侧的风险开关。
"""
import argparse
import sys
from pathlib import Path as _Path
import _path  # noqa: F401

import numpy as np
import pandas as pd

from chan_notes.eval.metrics import expectancy
from chan_notes.models import train


def sell_index(df: pd.DataFrame) -> dict:
    sells = df[df.is_buy == 0]
    return {c: np.sort(g["bar_idx"].to_numpy()) for c, g in sells.groupby("code")}


def recent_sell(code: str, t: int, n: int, idx: dict) -> bool:
    arr = idx.get(code)
    if arr is None or len(arr) == 0:
        return False
    lo = np.searchsorted(arr, t - n, side="right")
    hi = np.searchsorted(arr, t, side="right")
    return hi > lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="events_v3")
    ap.add_argument("--label", default="c_pol_dir_ret", help="c_pol_dir_ret=确认口径, pol_dir_ret=信号口径")
    ap.add_argument("--hold", type=int, default=0, help=">0 时用该持有期重算（配合 --stop）")
    ap.add_argument("--stop", type=float, default=0.0)
    args = ap.parse_args()

    df = train.load(args.name)
    idx = sell_index(df)
    if args.label.startswith("c_"):
        base = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
    else:
        base = df[df.is_buy == 1].copy()
    print(f"样本 {args.name}  口径 {args.label}  基数 {len(base)}")

    rows = []
    arrays = None
    if args.hold > 0:
        from chan_notes.core.policy import simulate_exit
        from chan_notes.eval.bars import for_policy, load_many
        arrays = {c: for_policy(a) for c, a in load_many(base.code.unique()).items()}
        idx_col = "conf_entry_idx" if args.label.startswith("c_") else "entry_idx"

    def rets_of(sub):
        if arrays is None:
            return sub[args.label].to_numpy()
        out = []
        for r in sub.itertuples():
            res = simulate_exit(arrays[r.code], int(getattr(r, idx_col)), int(r.is_buy), float(r.atr),
                                hold=args.hold, stop_atr=args.stop)
            if res is not None:
                out.append(res.ret if r.is_buy else -res.ret)
        return np.array(out)

    for n in (0, 4, 12, 24, 48, 96):
        if n == 0:
            sub = base
        else:
            sub = base[~base.apply(lambda r: recent_sell(r.code, int(r.bar_idx), n, idx), axis=1)]
        e = expectancy(rets_of(sub))
        rows.append({"no_sell_within": n, "n": e["n"], "win": e["win_rate"],
                     "R": e["payoff_R"], "E_bp": e["expectancy_bp"],
                     "breakeven_R": e["breakeven_R"], "kelly": e["kelly"],
                     "top5_share": e["top5pct_profit_share"]})
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    out.to_csv(_Path(__file__).resolve().parents[1] / "reports" / "sell_filter.csv", index=False)


if __name__ == "__main__":
    main()
