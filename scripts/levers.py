"""Lever analysis: win rate / payoff / expectancy / Kelly on a (hold, stop) grid."""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many
from chan_notes.eval.metrics import expectancy
from chan_notes.models import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="events_v3")
    ap.add_argument("--side", default="conf_buy", choices=["conf_buy", "buy", "all"])
    ap.add_argument("--holds", default="24,48,72,96")
    ap.add_argument("--stops", default="1.0,1.5,2.0,3.0")
    ap.add_argument("--trails", default="0")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    args = ap.parse_args()

    df = train.load(args.name)
    if args.start:
        df = df[df["event_time"] >= args.start]
    if args.end:
        df = df[df["event_time"] <= args.end]
    if args.side == "conf_buy":
        df = df[(df.is_buy == 1) & (df.conf_ok == 1)]
        idx_col = "conf_entry_idx"      # 必须用确认后的入场，否则是选择偏差
    elif args.side == "buy":
        df = df[df.is_buy == 1]
        idx_col = "entry_idx"
    else:
        idx_col = "entry_idx"
    arrays = {c: for_policy(a) for c, a in load_many(df.code.unique()).items()}
    holds = [int(x) for x in args.holds.split(",")]
    stops = [float(x) for x in args.stops.split(",")]
    trails = [float(x) for x in args.trails.split(",")]
    sides = [1] if args.side != "all" else [1, 0]

    rows = []
    for side in sides:
        sub = df[df.is_buy == side]
        for hold, stop, trail in itertools.product(holds, stops, trails):
            rets = []
            for r in sub.itertuples():
                res = simulate_exit(arrays[r.code], int(getattr(r, idx_col)), int(r.is_buy), float(r.atr),
                                    hold=hold, stop_atr=stop, trail_atr=trail)
                if res is not None:
                    rets.append(res.ret if r.is_buy else -res.ret)
            e = expectancy(rets)
            rows.append({"side": "buy" if side == 1 else "sell", "hold": hold, "stop": stop, "trail": trail,
                         "n": e["n"], "P": e["win_rate"], "R": e["payoff_R"],
                         "E_bp": e["expectancy_bp"], "R*": e["breakeven_R"], "kelly": e["kelly"]})
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))

    best_i = out["E_bp"].idxmax()
    b = out.loc[best_i]
    sub = df[df.is_buy == (1 if b["side"] == "buy" else 0)]
    rets = []
    for r in sub.itertuples():
        res = simulate_exit(arrays[r.code], int(getattr(r, idx_col)), int(r.is_buy), float(r.atr),
                            hold=int(b["hold"]), stop_atr=float(b["stop"]), trail_atr=float(b["trail"]))
        if res is not None:
            rets.append(res.ret if r.is_buy else -res.ret)
    e = expectancy(rets)
    W, L = e["avg_win_bp"], e["avg_loss_bp"]
    print(f"\nbest: side={b['side']} hold={int(b['hold'])} stop={b['stop']} P={e['win_rate']:.3f} "
          f"R={e['payoff_R']:.3f} E={e['expectancy_bp']:.1f}bp kelly={e['kelly']*100:.2f}%")
    print(f"avg win {W:.0f}bp / avg loss {L:.0f}bp")
    print("\ntarget E -> required P (R fixed) / required R (P fixed):")
    for t in (0, 20, 50, 100):
        need_p = (t + L) / (W + L) if (W + L) > 0 else np.nan
        need_r = (t + (1 - e["win_rate"]) * L) / (e["win_rate"] * L) if e["win_rate"] * L > 0 else np.nan
        print(f"  {t:4d}bp: P={need_p*100:5.1f}% (now {e['win_rate']*100:.1f}%) | R={need_r:.2f} (now {e['payoff_R']:.2f})")
    out.to_csv(Path(__file__).resolve().parents[1] / "reports" / "levers.csv", index=False)


if __name__ == "__main__":
    main()
