"""BTC 5m 宽区间失败突破→回归中线（候选 #2，预注册单一定义，一次 OOS）。

状态（5m 内自洽）：滚动 W=288 bar（24h）区间宽度 ≥ 5×ATR20（宽区间，喂得饱成本）。
事件：收盘越出区间（上破/下破，以越界前区间为准）→ K=6 bar 内收盘收回区间内 = 突破失败
→ 反向入场：上破失败做空 / 下破失败做多。
目标 = 越界前区间中线；止损 = 越界期间极值 ± 0.1ATR；持有上限 576 bar 超时收盘退出；
成交后 24 bar 冷却去重；同一越界过程只判一次（失败即平账，成功超时放弃）。
安慰剂 = 宽度合格随机 bar，复制真实事件止损/目标距离与方向，同口径退出。
先触止损优先（同 bar 双触保守）；末端不足持有期 = 删失剔除。
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
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

from chan_notes.data import store

COST = 0.0012
W = 288        # 区间回看（24h）
K = 6          # 失败判定窗口（30 分钟）
WIDTH_ATR = 5.0
BUF = 0.1
HOLD = 576     # 48h
COOL = 24


def load_bars():
    df = store.load("BTCUSDT", "5m")
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(df)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(20).mean().to_numpy()
    bh = pd.Series(high).rolling(W).max().shift(1).to_numpy()
    bl = pd.Series(low).rolling(W).min().shift(1).to_numpy()
    width_ok = (~np.isnan(atr)) & (~np.isnan(bh)) & ((bh - bl) >= WIDTH_ATR * atr)
    return high, low, close, atr, bh, bl, width_ok, n


def sim_exit_short(high, low, close, n, entry_i, entry_px, stop, target, hold):
    """做空：stop>entry 亏损，target<entry 盈利；先触止损优先。"""
    end = entry_i + hold
    for j in range(entry_i + 1, min(end, n)):
        if high[j] >= stop:
            return 1 - stop / entry_px - COST, j - entry_i, "stop"
        if low[j] <= target:
            return 1 - target / entry_px - COST, j - entry_i, "target"
    last = min(end, n - 1)
    if last < end:
        return np.nan, last - entry_i, "censored"
    return 1 - close[last] / entry_px - COST, last - entry_i, "time"


def sim_exit_long(high, low, close, n, entry_i, entry_px, stop, target, hold):
    end = entry_i + hold
    for j in range(entry_i + 1, min(end, n)):
        if low[j] <= stop:
            return stop / entry_px - 1 - COST, j - entry_i, "stop"
        if high[j] >= target:
            return target / entry_px - 1 - COST, j - entry_i, "target"
    last = min(end, n - 1)
    if last < end:
        return np.nan, last - entry_i, "censored"
    return close[last] / entry_px - 1 - COST, last - entry_i, "time"


def band_events(high, low, close, atr, bh, bl, n):
    events = []
    exc = None   # dict(side, start_i, ext, bh0, bl0)
    cool = -1
    for i in range(W, n):
        c = close[i]
        a = atr[i]
        if np.isnan(a) or np.isnan(bh[i]):
            continue
        if exc is None:
            if c > bh[i]:
                exc = {"side": 1, "start_i": i, "ext": high[i], "bh0": bh[i], "bl0": bl[i]}
            elif c < bl[i]:
                exc = {"side": -1, "start_i": i, "ext": low[i], "bh0": bh[i], "bl0": bl[i]}
            continue
        # 越界进行中
        exc["ext"] = max(exc["ext"], high[i]) if exc["side"] == 1 else min(exc["ext"], low[i])
        failed = c < exc["bh0"] if exc["side"] == 1 else c > exc["bl0"]
        if failed and i - exc["start_i"] <= K:
            if i > cool:
                a_i = a
                mid0 = (exc["bh0"] + exc["bl0"]) / 2
                if exc["side"] == 1:      # 上破失败 → 做空
                    entry_px = c
                    stop = exc["ext"] + BUF * a_i
                    events.append({"entry_i": i, "dir": -1, "entry_px": entry_px,
                                   "stop": stop, "target": mid0,
                                   "stop_dist": stop - entry_px, "target_dist": entry_px - mid0})
                else:                      # 下破失败 → 做多
                    entry_px = c
                    stop = exc["ext"] - BUF * a_i
                    events.append({"entry_i": i, "dir": 1, "entry_px": entry_px,
                                   "stop": stop, "target": mid0,
                                   "stop_dist": entry_px - stop, "target_dist": mid0 - entry_px})
                cool = i + COOL
            exc = None
        elif i - exc["start_i"] > K:
            exc = None   # 未失败 = 真突破，放弃
    return events


def main():
    t0 = time.time()
    high, low, close, atr, bh, bl, width_ok, n = load_bars()
    times = pd.to_datetime(store.load("BTCUSDT", "5m")["time"])
    years = times.dt.year.to_numpy()
    print(f"bars={n}  宽区间 bar 占比 {width_ok.mean() * 100:.1f}%（宽≥{WIDTH_ATR}×ATR20, W={W}）", flush=True)

    evs = band_events(high, low, close, atr, bh, bl, n)
    rows = []
    for e in evs:
        if e["dir"] == -1:
            ret, held, reason = sim_exit_short(high, low, close, n, e["entry_i"], e["entry_px"],
                                               e["stop"], e["target"], HOLD)
        else:
            ret, held, reason = sim_exit_long(high, low, close, n, e["entry_i"], e["entry_px"],
                                              e["stop"], e["target"], HOLD)
        e["ret"], e["held"], e["reason"] = ret, held, reason
        rows.append(e)
    dfe = pd.DataFrame(rows)
    cens = (dfe["reason"] == "censored").sum()
    dfe = dfe[dfe["reason"] != "censored"].copy()
    dfe["year"] = years[dfe["entry_i"].to_numpy()]
    for tag, sub in (("上破失败做空", dfe[dfe["dir"] == -1]), ("下破失败做多", dfe[dfe["dir"] == 1]),
                     ("合计", dfe)):
        r_is = sub[sub.year <= 2023]["ret"].to_numpy()
        r_oos = sub[sub.year >= 2024]["ret"].to_numpy()
        rr = sub["target_dist"] / sub["stop_dist"]
        print(f"\n[{tag}] n={len(sub)} 删失={cens if tag == '合计' else '--'} "
              f"中位R={rr.median():.2f} 中位持有={sub['held'].median():.0f}bar")
        print(f"  IS : E={r_is.mean() * 1e4:+7.1f}bp 胜率{(r_is > 0).mean() * 100:5.1f}% n={len(r_is)}")
        print(f"  OOS: E={r_oos.mean() * 1e4:+7.1f}bp 胜率{(r_oos > 0).mean() * 100:5.1f}% n={len(r_oos)}")

    # 安慰剂：宽度合格随机 bar，复制距离与方向
    rng = random.Random(42)
    ok_idx = np.where(width_ok)[0]
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
    print(f"\n安慰剂（宽度合格随机）: E={pl.mean() * 1e4:+.1f}bp 胜率{(pl > 0).mean() * 100:.1f}%")
    print(f"edge（策略-安慰剂）: OOS {dfe[dfe.year >= 2024]['ret'].mean() * 1e4 - pl.mean() * 1e4:+.1f}bp  "
          f"elapsed={time.time() - t0:.0f}s")
    dfe.to_parquet(ROOT / "data" / "events" / "range_fail.parquet", index=False)
    print("saved data/events/range_fail.parquet")


if __name__ == "__main__":
    main()
