"""BTC 5m 两腿回调（第二次恢复）事件电池——已确认定义的实施。

A 主状态（入场 bar 时刻全部因果已知）：
  日线方向向上（T-1 收盘 > T-61 收盘）+ ER60(T-1) ≥ er_thr + 5m 缠论线段向上（seg_dir==1）。
B 事件（因果 zigzag，幅值 = amp_atr × ATR20）：
  P1 上涨腿 L0→H1（腿高 ≥ min_leg × ATR）→ C1 回调至 L1 → R1 反弹至 H2 < H1（不创新高）
  → C2 回调至 L2 > L0（结构未失效）→ 收盘首次突破 H2 入场。
止损 = L2 - buf × ATR；目标 = H1 + 腿高（measured move）或入场 + 2R；
退出：先触止损优先（同 bar 双触保守），超时按收盘，末端不足持有期标 censored 剔除。
安慰剂：同状态随机 bar（同年），复制真实事件的止损/目标距离，同口径退出。
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
HOLD = 288          # 主口径持有：24h
BUF = 0.1           # 止损缓冲（×ATR）
ER_THR = 0.10
MIN_LEG_LIST = (1.0, 2.0)   # ×ATR
AMP_LIST = (0.8, 1.2)       # ×ATR
TARGET_LIST = ("mm", "2r")
random.seed(42)


def load_inputs():
    df = store.load("BTCUSDT", "5m")
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    n = len(df)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(20).mean().to_numpy()

    # 日线状态（T-1 已知）
    d = store.load("BTCUSDT", "d")
    dc = d["close"].to_numpy(float)
    chg = np.abs(np.diff(dc))
    er = np.full(len(dc), np.nan)
    for i in range(60, len(dc)):
        base = abs(dc[i] - dc[i - 60])
        vol = chg[i - 59: i + 1].sum()
        er[i] = base / vol if vol > 0 else 0.0
    dir_up = np.zeros(len(dc), dtype=bool)
    dir_up[61:] = dc[61:] > dc[:-61]
    # T-1 已知：把日线状态向后平移一天（用前一日值）
    er_known = np.roll(er, 1)
    dir_known = np.roll(dir_up, 1)
    er_known[0] = np.nan
    dir_known[0] = False
    dates = pd.to_datetime(d["time"]).dt.date.to_numpy()
    day_state = {dd: (dir_known[i], er_known[i]) for i, dd in enumerate(dates)}

    # 5m bar → 前一交易日状态
    bar_dates = pd.to_datetime(df["time"]).dt.date.to_numpy()
    daily_ok = np.zeros(n, dtype=bool)
    for i in range(n):
        st = day_state.get(bar_dates[i])
        if st is not None and st[0] and not np.isnan(st[1]) and st[1] >= ER_THR:
            daily_ok[i] = True

    # 5m 缠论线段状态
    st_path = ROOT / "data" / "state" / "btc5m_state.parquet"
    st = pd.read_parquet(st_path)
    assert len(st) == n, f"state rows {len(st)} != bars {n}"
    seg_up = (st["seg_dir"].to_numpy() == 1)

    state_ok = daily_ok & seg_up
    return df, high, low, close, atr, state_ok, daily_ok, seg_up, n


def zigzag_events(close, atr, state_ok, n, amp_atr, min_leg):
    """因果 zigzag + 两腿回调模式识别，返回事件列表。"""
    events = []
    armed = None  # (L2_idx, L2_px, H2_px, H1_px, leg_height)
    trend = 1
    ext_i, ext_p = 0, close[0]
    pivots = []  # (idx, price, kind) kind: 1=H, 0=L
    confirmed_this_bar = None

    for i in range(1, n):
        c = close[i]
        a = amp_atr * atr[i] if not np.isnan(atr[i]) else 0.0
        confirmed_this_bar = None
        if trend == 1:
            if c > ext_p:
                ext_i, ext_p = i, c
            elif a > 0 and ext_p - c >= a:
                pivots.append((ext_i, ext_p, 1))
                confirmed_this_bar = ("H", ext_i, ext_p)
                trend = -1
                ext_i, ext_p = i, c
        else:
            if c < ext_p:
                ext_i, ext_p = i, c
            elif a > 0 and c - ext_p >= a:
                pivots.append((ext_i, ext_p, 0))
                confirmed_this_bar = ("L", ext_i, ext_p)
                trend = 1
                ext_i, ext_p = i, c

        # 结构失效：收盘跌破 L0 → 解除武装
        if armed is not None and c < armed[1]:  # armed = (L2_idx, L2_px, H2, H1, leg)
            armed = None

        # 新 L 确认 → 尝试组装模式（L0,H1,L1,H2,L2）
        if confirmed_this_bar is not None and confirmed_this_bar[0] == "L":
            if len(pivots) >= 5:
                L2i, L2p, _ = pivots[-1]
                H2i, H2p, _ = pivots[-2]
                L1i, L1p, _ = pivots[-3]
                H1i, H1p, _ = pivots[-4]
                L0i, L0p, _ = pivots[-5]
                leg = H1p - L0p
                a_i = atr[i]
                if (H2p < H1p and L2p > L0p and L2p < H2p and L1p > L0p
                        and not np.isnan(a_i) and leg >= min_leg * a_i):
                    armed = (L2i, L2p, H2p, H1p, leg)

        # 武装中 + 突破 H2 → 入场
        if armed is not None and trend == 1 and c > armed[2]:
            L2i, L2p, H2p, H1p, leg = armed
            if state_ok[i]:
                a_i = atr[i]
                stop = L2p - BUF * a_i
                events.append({"entry_i": i, "entry_px": c, "stop": stop,
                               "mm_target": H1p + leg,
                               "r": (c - stop), "date": i})
            armed = None  # 无论是否入场都消耗本次武装（状态外 bar 不重复触发）
    return events


def sim_exit(high, low, close, n, entry_i, entry_px, stop, target, hold):
    """先触止损优先（保守）；超时收盘退出；末端不足持有期 = 删失。"""
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


def run_variant(high, low, close, atr, state_ok, n, amp_atr, min_leg, hold, target_kind):
    evs = zigzag_events(close, atr, state_ok, n, amp_atr, min_leg)
    return evs


def placebo(high, low, close, state_ok, n, real_df, hold, rng=random.Random(42)):
    """同状态随机 bar，复制真实事件的止损/目标距离。"""
    ok_idx = np.where(state_ok)[0]
    out = []
    if len(ok_idx) == 0 or len(real_df) == 0:
        return pd.DataFrame()
    by_year_cache = {}
    for _, e in real_df.iterrows():
        yr_candidates = by_year_cache.get(e["year"])
        if yr_candidates is None:
            # 每次都全扫太慢，用随机均匀抽样 ok_idx 即可（状态 bar 全期分布）
            yr_candidates = ok_idx
        j = int(yr_candidates[rng.randrange(len(yr_candidates))])
        entry_px = close[j]
        stop = entry_px - e["stop_dist"]
        target = entry_px + e["target_dist"]
        ret, held, reason = sim_exit(high, low, close, n, j, entry_px, stop, target, hold)
        out.append({"ret": ret, "held": held, "reason": reason})
    return pd.DataFrame(out)


def main():
    df, high, low, close, atr, state_ok, daily_ok, seg_up, n = load_inputs()
    times = pd.to_datetime(df["time"])
    years = times.dt.year.to_numpy()
    print(f"bars={n}  状态占比: daily_ok {daily_ok.mean() * 100:.1f}%  seg_up {seg_up.mean() * 100:.1f}%  交集 {state_ok.mean() * 100:.1f}%", flush=True)

    results = {}
    t0 = time.time()
    print("\n-- 模式格对比（mm / hold=288，IS 选格）--", flush=True)
    for amp in AMP_LIST:
        for ml in MIN_LEG_LIST:
            evs = zigzag_events(close, atr, state_ok, n, amp, ml)
            rr = []
            for e in evs:
                ret, held, reason = sim_exit(high, low, close, n, e["entry_i"], e["entry_px"],
                                             e["stop"], e["mm_target"], HOLD)
                if reason != "censored":
                    rr.append((years[e["entry_i"]], ret))
            r_is = np.array([r for y, r in rr if y <= 2023])
            r_oos = np.array([r for y, r in rr if y >= 2024])
            print(f"  amp={amp} min_leg={ml}: n={len(rr)} (IS={len(r_is)}, OOS={len(r_oos)}) "
                  f"IS E={r_is.mean() * 1e4:+7.1f}bp  OOS E={r_oos.mean() * 1e4:+7.1f}bp", flush=True)
            results[(amp, ml)] = (evs, r_is.mean() if len(r_is) else np.nan)

    # 主口径（IS E 最高的格）全表：目标类型 × 持有
    best = max(results, key=lambda k: results[k][1])
    print(f"\n选定模式格 amp={best[0]} min_leg={best[1]}（按 IS E，预注册规则）", flush=True)
    evs = results[best][0]
    for tk in TARGET_LIST:
        for hold in (96, HOLD, 576):
            rows = []
            for e in evs:
                dist_stop = e["entry_px"] - e["stop"]
                target = e["mm_target"] if tk == "mm" else e["entry_px"] + 2 * dist_stop
                ret, held, reason = sim_exit(high, low, close, n, e["entry_i"], e["entry_px"],
                                             e["stop"], target, hold)
                rows.append({**e, "ret": ret, "held": held, "reason": reason,
                             "stop_dist": dist_stop, "target_dist": target - e["entry_px"]})
            dfe = pd.DataFrame(rows)
            dfe = dfe[dfe["reason"] != "censored"]
            dfe["year"] = years[dfe["entry_i"].to_numpy()]
            r_is = dfe[dfe.year <= 2023]["ret"].to_numpy()
            r_oos = dfe[dfe.year >= 2024]["ret"].to_numpy()
            rr = dfe["target_dist"] / dfe["stop_dist"]
            pl = placebo(high, low, close, state_ok, n, dfe[dfe["reason"] != "censored"], hold)
            pl_e = pl["ret"].mean() * 1e4 if len(pl) else np.nan
            pl_w = (pl["ret"] > 0).mean() * 100 if len(pl) else np.nan
            def stat(r):
                if len(r) == 0:
                    return "        --"
                return f"{r.mean() * 1e4:+7.1f}bp {(r > 0).mean() * 100:5.1f}%"
            print(f"  目标={tk} hold={hold}: n={len(dfe)} "
                  f"IS[{stat(r_is)} n={len(r_is)}] OOS[{stat(r_oos)} n={len(r_oos)}] "
                  f"中位R={rr.median():.2f} 安慰剂E={pl_e:+.1f}bp/胜率{pl_w:.1f}%", flush=True)
            if tk == "mm" and hold == HOLD:
                dfe.to_parquet(ROOT / "data" / "events" / "two_leg_main.parquet", index=False)
                print("  saved data/events/two_leg_main.parquet", flush=True)

    # 退出原因分布（主口径）
    rows = []
    for e in evs:
        dist_stop = e["entry_px"] - e["stop"]
        ret, held, reason = sim_exit(high, low, close, n, e["entry_i"], e["entry_px"],
                                     e["stop"], e["mm_target"], HOLD)
        rows.append({**e, "ret": ret, "held": held, "reason": reason,
                     "stop_dist": dist_stop, "target_dist": e["mm_target"] - e["entry_px"]})
    dfe = pd.DataFrame(rows)
    dfe = dfe[dfe["reason"] != "censored"]
    dfe["year"] = years[dfe["entry_i"].to_numpy()]
    print(f"\n退出原因分布（mm/{HOLD}）: {dfe['reason'].value_counts().to_dict()}")
    print(f"R 中位 {((dfe['target_dist'] / dfe['stop_dist']).median()):.2f}  "
          f"中位持有 {dfe['held'].median():.0f} bars  elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
