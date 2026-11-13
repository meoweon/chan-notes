"""摆动点事件电池：机械层 IS/OOS 网格（买点做多 + 卖点做空）。
纪律：参数（amp）只允许看 IS(2020-2023) 选，OOS(2024-2026) 只评一次。
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.config import EVENTS
from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many

COST = 0.0012
b = for_policy(load_many(["BTCUSDT"], "30m")["BTCUSDT"])

for name in ("swing_b30_a1.5", "swing_b30_a2.5"):
    df = pd.read_parquet(EVENTS / f"{name}.parquet")
    df["year"] = pd.to_datetime(df.event_time).dt.year
    print(f"\n===== {name}  n={len(df)} =====")
    for side, label in ((1, "BUY"), (0, "SELL-short")):
        sub = df[df.is_buy == side]
        print(f"-- {label} (n={len(sub)}) --")
        print(f"{'hold':>5s} {'stop':>4s} | {'IS n':>5s} {'IS E':>8s} {'IS win':>6s} | {'OOS n':>5s} {'OOS E':>8s} {'OOS win':>7s}")
        for hold in (48, 144, 288, 576):
            for stop in (2.0, 3.0):
                outs = {}
                for tag, s in (("IS", sub[sub.year <= 2023]), ("OOS", sub[sub.year >= 2024])):
                    rets = []
                    for r in s.itertuples():
                        res = simulate_exit(b, int(r.conf_entry_idx), side, float(r.atr), hold=hold, stop_atr=stop, cost=COST)
                        if res is not None:
                            rets.append(res.ret)
                    rets = np.array(rets)
                    outs[tag] = (len(rets), rets.mean() * 1e4, (rets > 0).mean() * 100)
                (n1, e1, w1), (n2, e2, w2) = outs["IS"], outs["OOS"]
                print(f"{hold:>5d} {stop:>4.1f} | {n1:>5d} {e1:>+8.1f} {w1:>5.1f}% | {n2:>5d} {e2:>+8.1f} {w2:>6.1f}%")
