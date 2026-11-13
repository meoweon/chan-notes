"""BTC 5m 压缩结束突破跟进（候选 #3，预注册单一定义，一次 OOS）。

状态：越界前 W=288 bar 区间宽度/ATR20 ≤ thr（压缩；thr = IS 段（2020-23）该比值 20 分位，
在看任何 OOS 结果前冻结）。
事件：压缩状态下收盘首次越出区间沿 → 顺向入场（上破做多 / 下破做空，无确认步——
候选 #1 已证明确认步付时机税）。
目标 = 越界沿 + 区间高度（measured move）；止损 = 对沿 ∓ 0.1×ATR；
持有上限 576 bar；成交后 24 bar 冷却；先触止损优先；删失剔除。
安慰剂 = 压缩状态随机 bar，复制止损/目标距离与方向。
"""
import io
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["CHAN_NOTES_BARS"] = str(ROOT / "data" / "bars_crypto")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
# 不包 sys.stdout（同 two_leg_relaxed 教训）；-X utf8 保证编码。

import numpy as np
import pandas as pd

from range_fail_battery import (COST, W, load_bars, sim_exit_long, sim_exit_short)

BUF = 0.1
HOLD = 576
COOL = 24


def main():
    t0 = time.time()
    high, low, close, atr, bh, bl, _, n = load_bars()
    times = pd.to_datetime(__import__("chan_notes.data.store", fromlist=["store"]).load("BTCUSDT", "5m")["time"])
    years = times.dt.year.to_numpy()

    ratio = (bh - bl) / atr
    is_mask = years <= 2023
    valid = ~np.isnan(ratio)
    thr = float(np.nanquantile(ratio[is_mask & valid], 0.20))
    state = valid & (ratio <= thr)
    print(f"bars={n}  压缩阈值 thr={thr:.2f}×ATR20（IS 20 分位）  压缩 bar 占比 {state.mean() * 100:.1f}%", flush=True)

    events = []
    inside_prev = np.zeros(n, dtype=bool)
    inside_prev[W:] = (close[W - 1: n - 1] <= bh[W:]) & (close[W - 1: n - 1] >= bl[W:])
    cool = -1
    for i in range(W, n):
        c = close[i]
        if np.isnan(bh[i]) or np.isnan(atr[i]):
            continue
        if i < cool:
            continue
        broke_up = inside_prev[i] and c > bh[i]
        broke_dn = inside_prev[i] and c < bl[i]
        if not (broke_up or broke_dn) or not state[i]:
            continue
        w = bh[i] - bl[i]
        if broke_up:
            events.append({"entry_i": i, "dir": 1, "entry_px": c,
                           "stop": bl[i] - BUF * atr[i], "target": bh[i] + w,
                           "stop_dist": c - (bl[i] - BUF * atr[i]),
                           "target_dist": bh[i] + w - c})
        else:
            events.append({"entry_i": i, "dir": -1, "entry_px": c,
                           "stop": bh[i] + BUF * atr[i], "target": bl[i] - w,
                           "stop_dist": (bh[i] + BUF * atr[i]) - c,
                           "target_dist": c - (bl[i] - w)})
        cool = i + COOL

    rows = []
    for e in events:
        if e["dir"] == 1:
            ret, held, reason = sim_exit_long(high, low, close, n, e["entry_i"], e["entry_px"],
                                              e["stop"], e["target"], HOLD)
        else:
            ret, held, reason = sim_exit_short(high, low, close, n, e["entry_i"], e["entry_px"],
                                               e["stop"], e["target"], HOLD)
        rows.append({**e, "ret": ret, "held": held, "reason": reason})
    dfe = pd.DataFrame(rows)
    cens = (dfe["reason"] == "censored").sum() if len(dfe) else 0
    dfe = dfe[dfe["reason"] != "censored"].copy()
    dfe["year"] = years[dfe["entry_i"].to_numpy()]
    for tag, sub in (("上破做多", dfe[dfe["dir"] == 1]), ("下破做空", dfe[dfe["dir"] == -1]), ("合计", dfe)):
        if len(sub) == 0:
            continue
        r_is = sub[sub.year <= 2023]["ret"].to_numpy()
        r_oos = sub[sub.year >= 2024]["ret"].to_numpy()
        rr = sub["target_dist"] / sub["stop_dist"]
        print(f"\n[{tag}] n={len(sub)} 中位R={rr.median():.2f} 中位持有={sub['held'].median():.0f}bar")
        print(f"  IS : E={r_is.mean() * 1e4:+7.1f}bp 胜率{(r_is > 0).mean() * 100:5.1f}% n={len(r_is)}")
        print(f"  OOS: E={r_oos.mean() * 1e4:+7.1f}bp 胜率{(r_oos > 0).mean() * 100:5.1f}% n={len(r_oos)}")

    rng = random.Random(42)
    ok_idx = np.where(state)[0]
    pl = []
    for _, e in dfe.iterrows():
        j = int(ok_idx[rng.randrange(len(ok_idx))])
        entry_px = close[j]
        if e["dir"] == -1:
            ret, _, _ = sim_exit_short(high, low, close, n, j, entry_px,
                                       entry_px + e["stop_dist"], entry_px - e["target_dist"], HOLD)
        else:
            ret, _, _ = sim_exit_long(high, low, close, n, j, entry_px,
                                      entry_px - e["stop_dist"], entry_px + e["target_dist"], HOLD)
        pl.append(ret)
    pl = np.array(pl)
    print(f"\n删失={cens}  安慰剂（压缩状态随机）: E={pl.mean() * 1e4:+.1f}bp 胜率{(pl > 0).mean() * 100:.1f}%")
    print(f"毛收益（扣成本前）: OOS {dfe[dfe.year >= 2024]['ret'].mean() * 1e4 + COST * 1e4:+.1f}bp  "
          f"elapsed={time.time() - t0:.0f}s")
    dfe.to_parquet(ROOT / "data" / "events" / "compression_break.parquet", index=False)
    print("saved data/events/compression_break.parquet")


if __name__ == "__main__":
    main()
