"""Phase A tail test: does the edge survive removing the biggest winners?

For each (hold, stop) config, simulate per-trade returns for confirmed buys,
then remove the top 1%/5%/10% winners and recompute expectancy.
If the residual expectancy is ~0 or negative, the strategy's edge is tail luck.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.core.policy import simulate_exit
from chan_notes.eval.bars import for_policy, load_many
from chan_notes.eval.metrics import expectancy
from chan_notes.models import train


def simulate_returns(df, arrays, idx_col, hold, stop):
    rets, times, codes = [], [], []
    for r in df.itertuples():
        res = simulate_exit(arrays[r.code], int(getattr(r, idx_col)), int(r.is_buy), float(r.atr),
                            hold=hold, stop_atr=stop, trail_atr=0.0)
        if res is not None:
            rets.append(res.ret if r.is_buy else -res.ret)
            times.append(r.event_time)
            codes.append(r.code)
    return pd.DataFrame({"ret": rets, "event_time": times, "code": codes})


def tail_table(rets: np.ndarray, label: str) -> list[dict]:
    rows = []
    e0 = expectancy(rets)
    rows.append({"segment": label, "trim": "none", "n": e0["n"],
                 "P": e0["win_rate"], "R": e0["payoff_R"], "E_bp": e0["expectancy_bp"],
                 "kelly": e0["kelly"], "med_bp": float(np.median(rets)) * 1e4})
    srt = np.sort(rets)
    for q in (0.01, 0.05, 0.10):
        k = int(np.floor(len(srt) * q))
        if k == 0 or len(srt) - k < 30:
            continue
        rest = srt[: len(srt) - k]
        e = expectancy(rest)
        rows.append({"segment": label, "trim": f"-top{int(q*100)}%", "n": e["n"],
                     "P": e["win_rate"], "R": e["payoff_R"], "E_bp": e["expectancy_bp"],
                     "kelly": e["kelly"], "med_bp": float(np.median(rest)) * 1e4})
    return rows


def concentration(rets: np.ndarray, q: float) -> float:
    """Share of total net profit contributed by the top-q winners."""
    srt = np.sort(rets)
    k = int(np.floor(len(srt) * q))
    total = rets.sum()
    if total <= 0:
        return float("nan")
    return float(srt[-k:].sum() / total) if k else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="events_v4")
    ap.add_argument("--holds", default="96,144")
    ap.add_argument("--stops", default="3.0")
    args = ap.parse_args()

    df = train.load(args.name)
    df = df[(df.is_buy == 1) & (df.conf_ok == 1)]
    arrays = {c: for_policy(a) for c, a in load_many(df.code.unique()).items()}

    all_rows = []
    for hold in [int(x) for x in args.holds.split(",")]:
        for stop in [float(x) for x in args.stops.split(",")]:
            sim = simulate_returns(df, arrays, "conf_entry_idx", hold, stop)
            tag = f"hold={hold}/stop={stop}"
            rets = sim["ret"].values
            all_rows += tail_table(rets, tag)
            for q in (0.01, 0.05, 0.10):
                c = concentration(rets, q)
                print(f"[{tag}] top{int(q*100)}% winners carry {c*100:.0f}% of total net profit")
            # time split
            for seg, lo, hi in (("IS 2020-2023", "2020-01-01", "2023-12-31"),
                                ("OOS 2024-2026", "2024-01-01", "2026-12-31")):
                m = (sim["event_time"] >= lo) & (sim["event_time"] <= hi)
                if m.sum() >= 100:
                    all_rows += tail_table(sim.loc[m, "ret"].values, f"{tag} {seg}")

    out = pd.DataFrame(all_rows)
    print()
    print(out.round(3).to_string(index=False))
    out.to_csv(Path(__file__).resolve().parents[1] / "reports" / "tail_test.csv", index=False)
    print("\nsaved reports/tail_test.csv")


if __name__ == "__main__":
    main()
