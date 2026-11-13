"""Brooks 价格行为特征：全部只用 bar<=t 的数据。"""
import numpy as np


def _f(x, default=0.0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return v


def price_action_features(ind: dict, t: int) -> dict:
    o, h, l, c = ind["open"][t], ind["high"][t], ind["low"][t], ind["close"][t]
    f = {}
    rng = h - l
    body = abs(c - o)
    f["pa_body_ratio"] = body / rng if rng > 0 else 0.0
    f["pa_upper_shadow"] = (h - max(o, c)) / rng if rng > 0 else 0.0
    f["pa_lower_shadow"] = (min(o, c) - l) / rng if rng > 0 else 0.0
    f["pa_close_pos"] = (c - l) / rng if rng > 0 else 0.5
    f["pa_is_bull"] = 1.0 if c > o else (-1.0 if c < o else 0.0)

    if t >= 1:
        po, ph, pl, pc = ind["open"][t - 1], ind["high"][t - 1], ind["low"][t - 1], ind["close"][t - 1]
        f["pa_gap_up"] = 1.0 if l > ph else 0.0
        f["pa_gap_dn"] = 1.0 if h < pl else 0.0
        f["pa_inside"] = 1.0 if (h < ph and l > pl) else 0.0
        f["pa_outside"] = 1.0 if (h > ph and l < pl) else 0.0
        f["pa_ret1"] = (c - pc) / pc if pc > 0 else 0.0
        f["pa_range_expansion"] = rng / (ph - pl) if (ph - pl) > 0 else 1.0

        # 跟随柱（相对于前一根“信号柱”）
        prev_is_bull = ph > 0 and pc > po
        same_dir = (c > o and prev_is_bull) or (c < o and not prev_is_bull)
        f["pa_ft_dir_match"] = 1.0 if same_dir else 0.0
        f["pa_ft_break"] = 1.0 if ((c > o and h > ph) or (c < o and l < pl)) else 0.0
        f["pa_ft_body"] = f["pa_body_ratio"] if same_dir else -f["pa_body_ratio"]
        f["pa_ft_strength"] = 0.5 * f["pa_ft_dir_match"] + 0.3 * f["pa_ft_break"] + 0.2 * f["pa_body_ratio"]
    else:
        for k in ("pa_gap_up", "pa_gap_dn", "pa_inside", "pa_outside", "pa_ret1",
                  "pa_range_expansion", "pa_ft_dir_match", "pa_ft_break", "pa_ft_body", "pa_ft_strength"):
            f[k] = 0.0

    # 摆动结构（前 5 根，不含当前）
    if t >= 6:
        ph5 = ind["high"][t - 5:t].max()
        pl5 = ind["low"][t - 5:t].min()
        f["pa_hh"] = 1.0 if h > ph5 else 0.0
        f["pa_ll"] = 1.0 if l < pl5 else 0.0
        f["pa_hl"] = 1.0 if l > pl5 else 0.0
        f["pa_lh"] = 1.0 if h < ph5 else 0.0
    else:
        f.update({"pa_hh": 0.0, "pa_ll": 0.0, "pa_hl": 0.0, "pa_lh": 0.0})

    # 连续同向 bar
    streak, i = 0, t
    dirn = 1 if c > o else (-1 if c < o else 0)
    while i >= 0 and dirn != 0:
        di = 1 if ind["close"][i] > ind["open"][i] else (-1 if ind["close"][i] < ind["open"][i] else 0)
        if di != dirn:
            break
        streak += 1
        i -= 1
    f["pa_streak"] = float(dirn * min(streak, 10))

    atr = ind["atr14"][t]
    atr = atr if atr == atr and atr > 0 else max(rng, 1e-9)
    f["pa_bar_range_atr"] = rng / atr
    f["pa_atr_pct"] = atr / c if c > 0 else 0.0
    f["pa_std_pct"] = (ind["std20"][t] / c) if c > 0 and ind["std20"][t] == ind["std20"][t] else 0.0

    # 区间与突破
    hh20, ll20 = ind["hh20"][t], ind["ll20"][t]
    if hh20 == hh20 and ll20 == ll20 and hh20 > ll20:
        f["pa_range_pos20"] = _f((c - ll20) / (hh20 - ll20), 0.5)
        f["pa_dist_hh20"] = _f((hh20 - c) / c, 0.0)
        f["pa_dist_ll20"] = _f((c - ll20) / c, 0.0)
        f["pa_breakout_up"] = 1.0 if c > hh20 else 0.0
        f["pa_breakout_dn"] = 1.0 if c < ll20 else 0.0
        f["pa_fail_break_up"] = 1.0 if (h > hh20 and c < hh20) else 0.0
        f["pa_fail_break_dn"] = 1.0 if (l < ll20 and c > ll20) else 0.0
        f["pa_range_width20"] = _f((hh20 - ll20) / c, 0.0)
    else:
        f.update({"pa_range_pos20": 0.5, "pa_dist_hh20": 0.0, "pa_dist_ll20": 0.0,
                  "pa_breakout_up": 0.0, "pa_breakout_dn": 0.0,
                  "pa_fail_break_up": 0.0, "pa_fail_break_dn": 0.0, "pa_range_width20": 0.0})

    # 通道/趋势
    slope, r2 = ind["slope20"][t], ind["r2_20"][t]
    f["pa_slope20_norm"] = _f(slope / c, 0.0) if c > 0 else 0.0
    f["pa_r2_20"] = _f(r2, 0.0)
    trend = 0.0
    if f["pa_r2_20"] >= 0.55 and abs(f["pa_slope20_norm"]) > 1e-5:
        trend = 1.0 if f["pa_slope20_norm"] > 0 else -1.0
    f["pa_trend_dir"] = trend

    up, dn = ind["boll_up"][t], ind["boll_dn"][t]
    if up == up and dn == dn and up > dn:
        f["pa_boll_pos"] = _f((c - dn) / (up - dn), 0.5)
        f["pa_boll_width"] = _f((up - dn) / c, 0.0)
        f["pa_boll_break_up"] = 1.0 if c > up else 0.0
        f["pa_boll_break_dn"] = 1.0 if c < dn else 0.0
    else:
        f.update({"pa_boll_pos": 0.5, "pa_boll_width": 0.0, "pa_boll_break_up": 0.0, "pa_boll_break_dn": 0.0})

    ma5, ma10, ma20 = ind["ma5"][t], ind["ma10"][t], ind["ma20"][t]
    if ma5 == ma5 and ma10 == ma10 and ma20 == ma20:
        f["pa_ma_align"] = 1.0 if (ma5 > ma10 > ma20) else (-1.0 if (ma5 < ma10 < ma20) else 0.0)
        f["pa_price_vs_ma20"] = _f(c / ma20 - 1, 0.0) if ma20 > 0 else 0.0
        f["pa_ma5_vs_ma20"] = _f(ma5 / ma20 - 1, 0.0) if ma20 > 0 else 0.0
    else:
        f.update({"pa_ma_align": 0.0, "pa_price_vs_ma20": 0.0, "pa_ma5_vs_ma20": 0.0})

    for k, span in (("pa_ret5", 5), ("pa_ret20", 20), ("pa_ret60", 60)):
        if t >= span and ind["close"][t - span] > 0:
            f[k] = _f(c / ind["close"][t - span] - 1, 0.0)
        else:
            f[k] = 0.0

    # 动量指标
    dif, dea, mb = ind["dif"][t], ind["dea"][t], ind["macd_bar"][t]
    f["pa_macd_dif"] = _f(dif / c, 0.0) if c > 0 else 0.0
    f["pa_macd_dea"] = _f(dea / c, 0.0) if c > 0 else 0.0
    f["pa_macd_bar"] = _f(mb / c, 0.0) if c > 0 else 0.0
    cross = 0.0
    if t >= 1:
        pd_, pe_ = ind["dif"][t - 1], ind["dea"][t - 1]
        if pd_ == pd_ and pe_ == pe_:
            if dif > dea and pd_ <= pe_:
                cross = 1.0
            elif dif < dea and pd_ >= pe_:
                cross = -1.0
    f["pa_macd_cross"] = cross
    rsi = ind["rsi14"][t]
    f["pa_rsi"] = _f(rsi, 50.0)
    f["pa_rsi_ob"] = 1.0 if f["pa_rsi"] > 70 else 0.0
    f["pa_rsi_os"] = 1.0 if f["pa_rsi"] < 30 else 0.0

    # 量
    v, vma5, vma20 = ind["vol"][t], ind["vma5"][t], ind["vma20"][t]
    f["pa_vol_ratio5"] = _f(v / vma5, 1.0) if vma5 == vma5 and vma5 > 0 else 1.0
    f["pa_vol_ratio20"] = _f(v / vma20, 1.0) if vma20 == vma20 and vma20 > 0 else 1.0
    f["pa_vol_on_break"] = (1.0 if (f["pa_breakout_up"] or f["pa_breakout_dn"]) else 0.0) * min(f["pa_vol_ratio20"], 3.0)
    return f
