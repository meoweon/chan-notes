"""Brooks 形态特征：摆动点、双顶双底、楔形、突破回踩、测量移动、always-in。"""
import numpy as np


def _swings(ind, t, k: int = 3, lookback: int = 120):
    """返回 t 之前已确认的摆动高/低点 [(idx, price), ...]（升序）。"""
    lo = max(k, t - lookback)
    hi_idx, lo_idx = [], []
    for i in range(lo, t - k):
        h, l = ind["high"], ind["low"]
        if h[i] >= h[i - k:i + k + 1].max() and h[i] > h[i - 1]:
            hi_idx.append((i, float(h[i])))
        if l[i] <= l[i - k:i + k + 1].min() and l[i] < l[i - 1]:
            lo_idx.append((i, float(l[i])))
    return hi_idx, lo_idx


def pattern_features(ind: dict, t: int) -> dict:
    f = {}
    c = ind["close"][t]
    atr = ind["atr14"][t]
    atr = atr if atr == atr and atr > 0 else max(c * 0.01, 1e-9)

    highs, lows = _swings(ind, t)
    f["pt_n_swing_high"] = float(len(highs))
    f["pt_n_swing_low"] = float(len(lows))

    # 双顶/双底
    dt = db = 0.0
    if len(highs) >= 2:
        (i1, p1), (i2, p2) = highs[-2], highs[-1]
        mid_low = ind["low"][i1:i2 + 1].min() if i2 > i1 else p1
        if abs(p1 - p2) <= max(0.8 * atr, 0.01 * p1) and (i2 - i1) >= 5 and (p1 - mid_low) >= atr:
            dt = 1.0
            f["pt_dt_neck_dist"] = float((c - mid_low) / c)
    if len(lows) >= 2:
        (i1, p1), (i2, p2) = lows[-2], lows[-1]
        mid_high = ind["high"][i1:i2 + 1].max() if i2 > i1 else p1
        if abs(p1 - p2) <= max(0.8 * atr, 0.01 * p1) and (i2 - i1) >= 5 and (mid_high - p1) >= atr:
            db = 1.0
            f["pt_db_neck_dist"] = float((mid_high - c) / c)
    f["pt_double_top"] = dt
    f["pt_double_bottom"] = db
    f.setdefault("pt_dt_neck_dist", 0.0)
    f.setdefault("pt_db_neck_dist", 0.0)

    # 楔形：三次递进的推动，且后一推动幅度递减
    wedge_up = wedge_dn = 0.0
    if len(highs) >= 3:
        (i1, p1), (i2, p2), (i3, p3) = highs[-3], highs[-2], highs[-1]
        if p1 < p2 < p3:
            d1, d2 = p2 - p1, p3 - p2
            if d2 < d1 * 0.8 and (i3 - i1) >= 8:
                wedge_up = 1.0
    if len(lows) >= 3:
        (i1, p1), (i2, p2), (i3, p3) = lows[-3], lows[-2], lows[-1]
        if p1 > p2 > p3:
            d1, d2 = p1 - p2, p2 - p3
            if d2 < d1 * 0.8 and (i3 - i1) >= 8:
                wedge_dn = 1.0
    f["pt_wedge_up"] = wedge_up
    f["pt_wedge_dn"] = wedge_dn

    # always-in 方向：收盘 vs EMA20（用 ma20 代理）连续 5 根同侧
    ma20 = ind["ma20"]
    ai = 0.0
    if t >= 6 and ma20[t] == ma20[t]:
        above = sum(1 for i in range(t - 4, t + 1) if ind["close"][i] > ma20[i])
        below = sum(1 for i in range(t - 4, t + 1) if ind["close"][i] < ma20[i])
        if above == 5:
            ai = 1.0
        elif below == 5:
            ai = -1.0
    f["pt_always_in"] = ai

    # 突破回踩：最近 1-5 根内有 20 日新高/新低突破，当前 bar 回踩未破
    bo_pb_long = bo_pb_short = 0.0
    for j in range(max(t - 5, 21), t):
        if ind["hh20"][j] == ind["hh20"][j] and ind["close"][j] > ind["hh20"][j]:
            level = ind["hh20"][j]
            if ind["low"][t] <= level + 0.5 * atr and c > level and c > ind["open"][t]:
                bo_pb_long = 1.0
            break
    for j in range(max(t - 5, 21), t):
        if ind["ll20"][j] == ind["ll20"][j] and ind["close"][j] < ind["ll20"][j]:
            level = ind["ll20"][j]
            if ind["high"][t] >= level - 0.5 * atr and c < level and c < ind["open"][t]:
                bo_pb_short = 1.0
            break
    f["pt_bo_pb_long"] = bo_pb_long
    f["pt_bo_pb_short"] = bo_pb_short

    # 测量移动空间：以最近摆动区间高度为目标
    room_up = room_dn = 0.0
    if len(highs) >= 1 and len(lows) >= 1:
        (ih, ph) = highs[-1]
        (il, pl) = lows[-1]
        height = abs(ph - pl)
        target_up = max(ph, c) + height
        target_dn = min(pl, c) - height
        room_up = (target_up - c) / c
        room_dn = (c - target_dn) / c
    f["pt_mm_room_up"] = float(room_up)
    f["pt_mm_room_dn"] = float(room_dn)

    # 尖峰 / 通道 / 窄幅震荡
    r3 = ind["high"][t - 2:t + 1].max() - ind["low"][t - 2:t + 1].min() if t >= 2 else 0.0
    f["pt_spike"] = 1.0 if r3 > 2.5 * atr else 0.0
    f["pt_tight_range"] = 1.0 if (ind["range20"][t] == ind["range20"][t]
                                  and ind["range20"][t] * c < 3.0 * atr) else 0.0
    return f
