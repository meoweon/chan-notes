"""安慰剂对照：同样的退出规则，把入场点随机挪到附近，看是否还有同样的期望。

如果随机入场的 E 与真实信号相当，说明收益来自 beta/持有期，不是信号。
"""
import argparse
import sys
from pathlib import Path as _Path
import _path  # noqa: F401

import numpy as np

from chan_notes.core.policy import simulate_exit
from chan_notes.data import store
from chan_notes.eval.bars import for_policy, load_many
from chan_notes.eval.metrics import expectancy
from chan_notes.features import indicators
from chan_notes.models import train

rng = np.random.default_rng(42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="events_v3")
    ap.add_argument("--hold", type=int, default=144)
    ap.add_argument("--holds", default="", help="逗号分隔，给定时逐个持有期做对照")
    ap.add_argument("--stop", type=float, default=3.0)
    ap.add_argument("--jitter", type=int, default=120, help="随机入场相对真实入场的最大偏移 bar")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--reps", type=int, default=3, help="每个事件抽几个随机对照")
    args = ap.parse_args()

    df = train.load(args.name)
    df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
    if args.start:
        df = df[df["event_time"] >= args.start]
    if args.end:
        df = df[df["event_time"] <= args.end]
    codes = df.code.unique()
    arrays = {c: for_policy(a) for c, a in load_many(codes).items()}
    inds = {c: indicators.precompute(store.load(c)) for c in codes}

    real, ctrl = [], []
    holds = [int(x) for x in args.holds.split(",")] if args.holds else [args.hold]
    for hold in holds:
        real = []
        ctrl = []
        for r in df.itertuples():
            b = arrays[r.code]
            n = len(b["close"])
            i0 = int(r.conf_entry_idx)
            res = simulate_exit(b, i0, 1, float(r.atr), hold=hold, stop_atr=args.stop)
            if res:
                real.append(res.ret)
            for _ in range(args.reps):
                off = int(rng.integers(-args.jitter, args.jitter + 1))
                j = min(max(i0 + off, 300), n - 2)
                a = inds[r.code]["atr14"][j]
                if not (a == a and a > 0):
                    continue
                rr = simulate_exit(b, j, 1, float(a), hold=hold, stop_atr=args.stop)
                if rr:
                    ctrl.append(rr.ret)
        er, ec = expectancy(real), expectancy(ctrl)
        print(f"hold={hold:3d}  signal E={er['expectancy_bp']:7.1f}bp P={er['win_rate']:.3f} R={er['payoff_R']:.2f} | "
              f"placebo E={ec['expectancy_bp']:6.1f}bp P={ec['win_rate']:.3f} R={ec['payoff_R']:.2f} | "
              f"edge={er['expectancy_bp'] - ec['expectancy_bp']:6.1f}bp", flush=True)


if __name__ == "__main__":
    main()
