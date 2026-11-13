"""Brooks 结构特征：摆动点 / regime / bar counting / 突破强度 / 旗形与区间。

纯函数，保持"只用 ≤ 当前 bar"的性质：
- 摆动点要求右侧 confirm 根确认，未确认的候选不输出；
- bar counting 用"最近一个已确认摆动低点"作为起点，逐根统计创新高的次数；
- 突破强度只看突破棒本身，跟随棒交给下游（避免未来数据）。
"""
import numpy as np

CONFIRM = 3


def _f(x, d=0.0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return d
    return d if v != v or v in (float("inf"), float("-inf")) else v


def detect_swings(ind: dict, t: int, confirm: int = CONFIRM, lookback: int = 150):
    """返回 t 时刻已确认的摆动点 [(idx, price, kind)]，kind ∈ {1: 高点, -1: 低点}。"""
    hi, lo = ind["high"], ind["low"]
    out = []
    start = max(confirm, t - lookback)
    for i in range(start, t - confirm + 1):
        left_h, right_h = hi[i - confirm:i], hi[i + 1:i + 1 + confirm]
        left_l, right_l = lo[i - confirm:i], lo[i + 1:i + 1 + confirm]
        if len(right_h) < confirm:
            continue
        if hi[i] > left_h.max() and hi[i] > right_h.max():
            out.append((i, float(hi[i]), 1))
        elif lo[i] < left_l.min() and lo[i] < right_l.min():
            out.append((i, float(lo[i]), -1))
    return out


def swing_features(ind: dict, t: int) -> dict:
    sw = detect_swings(ind, t)
    f = {"sw_n": float(len(sw)), "sw_bull_ratio": 0.5, "sw_label": 0.0,
         "sw_dist_h": 0.0, "sw_dist_l": 0.0, "sw_bars_since": 0.0}
    if not sw:
        return f
    c = ind["close"][t]
    highs = [s for s in sw if s[2] == 1][-6:]
    lows = [s for s in sw if s[2] == -1][-6:]
    hh = sum(1 for a, b in zip(highs, highs[1:]) if b[1] > a[1])
    lh = sum(1 for a, b in zip(highs, highs[1:]) if b[1] <= a[1])
    hl = sum(1 for a, b in zip(lows, lows[1:]) if b[1] > a[1])
    ll = sum(1 for a, b in zip(lows, lows[1:]) if b[1] <= a[1])
    total = hh + lh + hl + ll
    bull = (hh + hl) / total if total else 0.5
    f["sw_bull_ratio"] = bull
    f["sw_bars_since"] = float(t - sw[-1][0])
    if highs:
        f["sw_dist_h"] = _f(highs[-1][1] / c - 1) if c > 0 else 0.0
    if lows:
        f["sw_dist_l"] = _f(c / lows[-1][1] - 1) if lows[-1][1] > 0 else 0.0

    ma20, ma60 = ind["ma20"][t], ind["ma60"][t]
    ma_bull = (ma20 == ma20 and ma60 == ma60 and ma20 > ma60)
    if bull >= 0.6 and ma_bull:
        f["sw_label"] = 1.0
    elif bull <= 0.4 and ma20 == ma20 and ma60 == ma60 and ma20 < ma60:
        f["sw_label"] = -1.0
    return f


def regime_features(ind: dict, t: int) -> dict:
    """趋势 / 区间 / 过渡 三态，外加质量分与区间宽度。"""
    c = ind["close"][t]
    ma20, ma60 = ind["ma20"][t], ind["ma60"][t]
    hh20, ll20 = ind["hh20"][t], ind["ll20"][t]
    atr = ind["atr14"][t]
    f = {"rg_label": 0.0, "rg_bull_ratio": 0.5, "rg_ma_bull": 0.0, "rg_range_atr": 0.0,
         "rg_quality": 0.0}
    if ma20 != ma20 or ma60 != ma60:
        return f
    sw = detect_swings(ind, t, lookback=100)
    highs = [s for s in sw if s[2] == 1][-5:]
    lows = [s for s in sw if s[2] == -1][-5:]
    hh = sum(1 for a, b in zip(highs, highs[1:]) if b[1] > a[1])
    lh = sum(1 for a, b in zip(highs, highs[1:]) if b[1] <= a[1])
    hl = sum(1 for a, b in zip(lows, lows[1:]) if b[1] > a[1])
    ll = sum(1 for a, b in zip(lows, lows[1:]) if b[1] <= a[1])
    total = hh + lh + hl + ll
    bull = (hh + hl) / total if total else 0.5
    f["rg_bull_ratio"] = bull
    f["rg_ma_bull"] = 1.0 if ma20 > ma60 else 0.0
    if hh20 == hh20 and ll20 == ll20 and atr > 0:
        f["rg_range_atr"] = float((hh20 - ll20) / atr)
    if bull >= 0.6 and ma20 > ma60:
        f["rg_label"] = 1.0
        f["rg_quality"] = bull
    elif bull <= 0.4 and ma20 < ma60:
        f["rg_label"] = -1.0
        f["rg_quality"] = 1 - bull
    else:
        f["rg_label"] = 0.0
        f["rg_quality"] = 0.5
    return f


def bar_count_features(ind: dict, t: int) -> dict:
    """H1/H2/L1/L2 近似计数（Brooks 定义的对前一根高低点的突破次数）。

    以最近一个已确认摆动低点（数 H）或高点（数 L）为锚点，逐根统计：
    - H 计数：high[i] > high[i-1] 的累计次数；最后一次计数距当前的 bar 数；
    - L 计数：low[i] < low[i-1] 的累计次数。
    这是工程近似：Brooks 原始定义还要求"回落/反弹上下文"，本版只用锚点近似。
    """
    f = {"bc_h": 0.0, "bc_l": 0.0, "bc_signal_age": 0.0, "bc_side": 0.0}
    sw = detect_swings(ind, t, lookback=80)
    if not sw:
        return f
    anchor_l = next((s[0] for s in reversed(sw) if s[2] == -1), None)
    anchor_h = next((s[0] for s in reversed(sw) if s[2] == 1), None)
    hi, lo = ind["high"], ind["low"]
    if anchor_l is not None:
        h = 0
        last = anchor_l
        for i in range(anchor_l + 1, t + 1):
            if hi[i] > hi[i - 1]:
                h += 1
                last = i
        f["bc_h"] = float(h)
        f["bc_signal_age"] = float(t - last)
    if anchor_h is not None:
        l = 0
        for i in range(anchor_h + 1, t + 1):
            if lo[i] < lo[i - 1]:
                l += 1
        f["bc_l"] = float(l)
    f["bc_side"] = 1.0 if f["bc_h"] >= f["bc_l"] else -1.0
    return f


def breakout_strength_features(ind: dict, t: int) -> dict:
    """R05 口径的突破强度：大棒（≥1.2×近 14 根均幅）+ 极端收盘（≥0.7）。"""
    f = {"bsw_level": 0.0, "bsw_big": 0.0, "bsw_extreme": 0.0, "bsw_crossed": 0.0}
    hh20 = ind["hh20"][t]
    c, o, h, l = ind["close"][t], ind["open"][t], ind["high"][t], ind["low"][t]
    if hh20 != hh20:
        return f
    crossed_up = c > hh20
    ll20 = ind["ll20"][t]
    crossed_dn = ll20 == ll20 and c < ll20
    if not (crossed_up or crossed_dn):
        return f
    rng = h - l
    avg = ind["high"][max(t - 14, 0):t] - ind["low"][max(t - 14, 0):t]
    avg_range = float(avg.mean()) if len(avg) else 0.0
    big = rng >= 1.2 * avg_range if avg_range > 0 else False
    pos = (c - l) / rng if rng > 0 else 0.5
    extreme = (pos >= 0.7) if crossed_up else (pos <= 0.3)
    f["bsw_big"] = 1.0 if big else 0.0
    f["bsw_extreme"] = 1.0 if extreme else 0.0
    f["bsw_crossed"] = 1.0 if crossed_up else -1.0
    f["bsw_level"] = 3.0 if (big and extreme) else (2.0 if big else 1.0)
    return f


def pattern2_features(ind: dict, t: int) -> dict:
    """final flag（紧凑旗形）+ range fade（区间边缘）+ 楔形质量。"""
    f = {"ff_flag": 0.0, "rf_long": 0.0, "rf_short": 0.0, "wg_quality": 0.0}
    if t < 26:
        return f
    h, l, o, c = ind["high"], ind["low"], ind["open"], ind["close"]
    rng5 = (h[t - 4:t + 1] - l[t - 4:t + 1])
    prior = (h[t - 24:t - 4] - l[t - 24:t - 4])
    prior_avg = float(prior.mean()) if len(prior) else 0.0
    bodies = np.abs(c[t - 4:t + 1] - o[t - 4:t + 1])
    if prior_avg > 0:
        compact = bool((rng5 <= 0.7 * prior_avg).all())
        small = bool((bodies <= 0.5 * rng5).all())
        f["ff_flag"] = 1.0 if (compact and small) else 0.0

    atr = ind["atr14"][t]
    hh20, ll20 = ind["hh20"][t], ind["ll20"][t]
    if hh20 == hh20 and ll20 == ll20 and atr > 0 and (hh20 - ll20) >= 2.5 * atr:
        pos = (c[t] - ll20) / (hh20 - ll20)
        f["rf_long"] = 1.0 if pos <= 0.25 else 0.0
        f["rf_short"] = 1.0 if pos >= 0.75 else 0.0

    sw = detect_swings(ind, t, lookback=100)
    highs = [s for s in sw if s[2] == 1][-3:]
    lows = [s for s in sw if s[2] == -1][-3:]
    q = 0.0
    if len(highs) == 3 and highs[0][1] < highs[1][1] < highs[2][1]:
        d1, d2 = highs[1][1] - highs[0][1], highs[2][1] - highs[1][1]
        q = max(q, float(d2 / d1 - 1) if d1 > 0 else 0.0)   # 负值=推力衰减
    if len(lows) == 3 and lows[0][1] > lows[1][1] > lows[2][1]:
        d1, d2 = lows[0][1] - lows[1][1], lows[1][1] - lows[2][1]
        q = max(q, float(d2 / d1 - 1) if d1 > 0 else 0.0)
    f["wg_quality"] = q
    return f
