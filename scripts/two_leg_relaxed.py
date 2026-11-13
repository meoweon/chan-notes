"""两腿回调 v1.1 放宽重测（一次性预注册，不做第二遍）。

相对 v1 只改两处（其余全部不动）：
①入场提前：L2 确认当根收盘入场（去掉突破 H2 的确认步）；
②止损放宽：L0 - 0.1ATR（结构失效位，代替 L2）。
网格同 v1（amp×min_leg，IS 选格）、mm 目标、hold=288、安慰剂同口径。
目的：检验 v1 的"时机税"假说——若损失来自确认入场买在短线顶部，提前入场应回收安慰剂漂移。
"""
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["CHAN_NOTES_BARS"] = str(ROOT / "data" / "bars_crypto")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# 注意：不要在此处包 sys.stdout——导入 two_leg_battery 时它会再包一层，双层包装会关闭底层流。
# 统一用 -X utf8 运行保证编码。

import numpy as np
import pandas as pd

from two_leg_battery import AMP_LIST, BUF, COST, HOLD, MIN_LEG_LIST, load_inputs, placebo, sim_exit


def zigzag_relaxed(close, atr, state_ok, n, amp_atr, min_leg):
    """v1 同构几何条件，但入场=L2 确认当根、止损=L0-0.1ATR。"""
    events = []
    trend = 1
    ext_i, ext_p = 0, close[0]
    pivots = []
    for i in range(1, n):
        c = close[i]
        a = amp_atr * atr[i] if not np.isnan(atr[i]) else 0.0
        confirmed = None
        if trend == 1:
            if c > ext_p:
                ext_i, ext_p = i, c
            elif a > 0 and ext_p - c >= a:
                pivots.append((ext_i, ext_p, 1))
                confirmed = (ext_i, ext_p)
                trend = -1
                ext_i, ext_p = i, c
        else:
            if c < ext_p:
                ext_i, ext_p = i, c
            elif a > 0 and c - ext_p >= a:
                pivots.append((ext_i, ext_p, 0))
                confirmed = (ext_i, ext_p)
                trend = 1
                ext_i, ext_p = i, c

        if confirmed is not None and pivots[-1][2] == 0 and len(pivots) >= 5:
            L2i, L2p = confirmed
            _, H2p, _ = pivots[-2]
            _, L1p, _ = pivots[-3]
            _, H1p, _ = pivots[-4]
            _, L0p, _ = pivots[-5]
            leg = H1p - L0p
            a_i = atr[i]
            if (H2p < H1p and L2p > L0p and L2p < H2p and L1p > L0p
                    and not np.isnan(a_i) and leg >= min_leg * a_i and state_ok[i]):
                events.append({"entry_i": i, "entry_px": c,
                               "stop": L0p - BUF * a_i,
                               "mm_target": H1p + leg})
    return events


def main():
    df, high, low, close, atr, state_ok, daily_ok, seg_up, n = load_inputs()
    times = pd.to_datetime(df["time"])
    years = times.dt.year.to_numpy()
    print(f"bars={n}  状态交集 {state_ok.mean() * 100:.1f}%（与 v1 相同）", flush=True)

    best, best_e = None, -np.inf
    grids = {}
    for amp in AMP_LIST:
        for ml in MIN_LEG_LIST:
            evs = zigzag_relaxed(close, atr, state_ok, n, amp, ml)
            rows = []
            for e in evs:
                ret, held, reason = sim_exit(high, low, close, n, e["entry_i"], e["entry_px"],
                                             e["stop"], e["mm_target"], HOLD)
                if reason != "censored":
                    rows.append((years[e["entry_i"]], ret, e))
            r_is = np.array([r for y, r, _ in rows if y <= 2023])
            r_oos = np.array([r for y, r, _ in rows if y >= 2024])
            e_is = r_is.mean() * 1e4 if len(r_is) else np.nan
            e_oos = r_oos.mean() * 1e4 if len(r_oos) else np.nan
            print(f"  amp={amp} min_leg={ml}: n={len(rows)} (IS={len(r_is)}, OOS={len(r_oos)}) "
                  f"IS E={e_is:+7.1f}bp  OOS E={e_oos:+7.1f}bp", flush=True)
            grids[(amp, ml)] = rows
            if e_is > best_e:
                best, best_e = (amp, ml), e_is

    print(f"\n选定格 amp={best[0]} min_leg={best[1]}（IS E={best_e:+.1f}bp，预注册规则）", flush=True)
    rows = grids[best]
    dfe = pd.DataFrame([{"entry_i": e["entry_i"], "entry_px": e["entry_px"], "ret": r,
                         "stop_dist": e["entry_px"] - e["stop"],
                         "target_dist": e["mm_target"] - e["entry_px"], "year": y}
                        for y, r, e in rows])
    r_is = dfe[dfe.year <= 2023]["ret"].to_numpy()
    r_oos = dfe[dfe.year >= 2024]["ret"].to_numpy()
    rr = dfe["target_dist"] / dfe["stop_dist"]
    pl = placebo(high, low, close, state_ok, n, dfe, HOLD)
    print(f"主口径 mm/hold={HOLD}: n={len(dfe)}  "
          f"IS[{r_is.mean() * 1e4:+7.1f}bp {(r_is > 0).mean() * 100:5.1f}% n={len(r_is)}]  "
          f"OOS[{r_oos.mean() * 1e4:+7.1f}bp {(r_oos > 0).mean() * 100:5.1f}% n={len(r_oos)}]")
    print(f"中位R={rr.median():.2f}  安慰剂E={pl['ret'].mean() * 1e4:+.1f}bp/"
          f"胜率{(pl['ret'] > 0).mean() * 100:.1f}%  cost={COST}")
    dfe.to_parquet(ROOT / "data" / "events" / "two_leg_relaxed.parquet", index=False)
    print("saved data/events/two_leg_relaxed.parquet")


if __name__ == "__main__":
    main()
