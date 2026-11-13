"""缠论结构特征：笔 / 线段 / 中枢 / 买卖点，取决策时(t)的可见状态。"""
import numpy as np

from Common.CEnum import BSP_TYPE


def _f(x, d=0.0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return d
    if v != v or v in (float("inf"), float("-inf")):
        return d
    return v


BSP_TYPES = [BSP_TYPE.T1, BSP_TYPE.T1P, BSP_TYPE.T2, BSP_TYPE.T2S, BSP_TYPE.T3A, BSP_TYPE.T3B]


def structure_features(lv, klu, bsp=None) -> dict:
    f = {}
    bis = lv.bi_list
    n_bi = len(bis)
    f["ch_n_bi"] = float(n_bi)
    f["ch_n_sure_bi"] = float(sum(1 for b in bis if b.is_sure))

    bi = bis[-1] if n_bi else None
    if bi is not None:
        klc_lst = list(bi.klc_lst)
        f["ch_bi_dir"] = 1.0 if bi.is_up() else -1.0
        f["ch_bi_sure"] = 1.0 if bi.is_sure else 0.0
        f["ch_bi_klc_cnt"] = float(len(klc_lst))
        f["ch_bi_klu_cnt"] = float(bi.get_klu_cnt())
        f["ch_bi_amp"] = _f(bi.amp())
        f["ch_bi_slope"] = _f(bi.amp() / max(bi.get_klu_cnt(), 1))
        # 当前 K 线在笔中的位置
        pos = 0
        for i, k in enumerate(klc_lst):
            if k is klu.klc:
                pos = i
                break
        f["ch_bi_pos"] = pos / max(len(klc_lst) - 1, 1)
        f["ch_bi_idx"] = float(bi.idx)

        # 同向笔连续数、与前一笔的幅度比
        streak, i = 0, n_bi - 1
        d = bi.is_up()
        while i >= 0 and bis[i].is_up() == d:
            streak += 1
            i -= 1
        f["ch_bi_dir_streak"] = float(streak)
        if n_bi >= 2:
            prev = bis[-2]
            f["ch_bi_amp_prev"] = _f(prev.amp())
            pa = abs(prev.amp())
            f["ch_bi_amp_ratio"] = _f(abs(bi.amp()) / pa, 1.0) if pa > 0 else 1.0
        else:
            f["ch_bi_amp_prev"] = 0.0
            f["ch_bi_amp_ratio"] = 1.0
    else:
        for k in ("ch_bi_dir", "ch_bi_sure", "ch_bi_klc_cnt", "ch_bi_klu_cnt", "ch_bi_amp",
                  "ch_bi_slope", "ch_bi_pos", "ch_bi_idx", "ch_bi_dir_streak", "ch_bi_amp_prev"):
            f[k] = 0.0
        f["ch_bi_amp_ratio"] = 1.0

    # 线段
    segs = lv.seg_list.lst
    seg = segs[-1] if segs else None
    if seg is not None:
        f["ch_seg_dir"] = 1.0 if seg.is_up() else -1.0
        f["ch_seg_sure"] = 1.0 if seg.is_sure else 0.0
        f["ch_seg_bi_cnt"] = float(len(seg.bi_list))
        f["ch_seg_amp"] = _f(seg.amp())
        f["ch_seg_klu_cnt"] = float(seg.get_klu_cnt())
        if bi is not None and bi in seg.bi_list:
            f["ch_bi_in_seg_pos"] = seg.bi_list.index(bi) / max(len(seg.bi_list) - 1, 1)
        else:
            f["ch_bi_in_seg_pos"] = 0.0
        f["ch_seg_n_zs"] = float(len(seg.zs_lst))
        f["ch_seg_has_trendline"] = 1.0 if getattr(seg, "support_trend_line", None) is not None else 0.0
    else:
        for k in ("ch_seg_dir", "ch_seg_sure", "ch_seg_bi_cnt", "ch_seg_amp", "ch_seg_klu_cnt",
                  "ch_bi_in_seg_pos", "ch_seg_n_zs", "ch_seg_has_trendline"):
            f[k] = 0.0

    # 中枢
    zss = lv.zs_list.zs_lst
    f["ch_n_zs"] = float(len(zss))
    zs = zss[-1] if zss else None
    if zs is not None and zs.high > zs.low:
        c = klu.close
        f["ch_zs_high"] = _f(zs.high)
        f["ch_zs_low"] = _f(zs.low)
        f["ch_zs_width"] = _f((zs.high - zs.low) / zs.low)
        f["ch_zs_bi_cnt"] = float(len(zs.bi_lst))
        f["ch_zs_is_one_bi"] = 1.0 if zs.is_one_bi_zs() else 0.0
        f["ch_zs_pos"] = _f((c - zs.low) / (zs.high - zs.low), 0.5)
        f["ch_zs_dist_high"] = _f((zs.high - c) / c)
        f["ch_zs_dist_low"] = _f((c - zs.low) / c)
        f["ch_price_vs_zs_mid"] = _f((c - zs.mid) / zs.mid) if zs.mid > 0 else 0.0
        f["ch_in_zs"] = 1.0 if zs.low <= c <= zs.high else 0.0
        # 最近一根笔是否已离开中枢
        if bi is not None:
            endv = bi.get_end_val()
            f["ch_bi_leaves_zs"] = 1.0 if (endv > zs.high or endv < zs.low) else 0.0
        else:
            f["ch_bi_leaves_zs"] = 0.0
        f["ch_zs_peak_width"] = _f((zs.peak_high - zs.peak_low) / zs.peak_low) if zs.peak_low > 0 else 0.0
        f["ch_bars_since_zs_end"] = float(max(klu.idx - zs.end_bi.get_end_klu().idx, 0))
    else:
        for k in ("ch_zs_high", "ch_zs_low", "ch_zs_width", "ch_zs_bi_cnt", "ch_zs_is_one_bi",
                  "ch_zs_pos", "ch_zs_dist_high", "ch_zs_dist_low", "ch_price_vs_zs_mid", "ch_in_zs",
                  "ch_bi_leaves_zs", "ch_zs_peak_width", "ch_bars_since_zs_end"):
            f[k] = 0.0

    # 买卖点（只在事件 bar 上有值）；固定列，不随类型增减
    for bt in BSP_TYPES:
        f[f"ch_bsp_{bt.value}"] = 0.0
    f["ch_bsp_is_buy"] = 0.0
    f["ch_bsp_div"] = 0.0
    f["ch_bsp_zs_cnt"] = 0.0
    f["ch_bsp_bi_amp"] = 0.0
    fixed_extra = ("bsp2_retrace_rate", "bsp2_break_bi_amp", "bsp2_bi_amp",
                   "bsp2s_retrace_rate", "bsp2s_break_bi_amp", "bsp2s_bi_amp", "bsp2s_lv",
                   "bsp3_zs_height", "bsp3_bi_amp")
    for k in fixed_extra:
        f[f"ch_{k}"] = 0.0
    if bsp is not None:
        for bt in bsp.type:
            f[f"ch_bsp_{bt.value}"] = 1.0
        f["ch_bsp_is_buy"] = 1.0 if bsp.is_buy else 0.0
        feats = dict(bsp.features.items()) if hasattr(bsp, "features") else {}
        f["ch_bsp_div"] = _f(feats.get("divergence_rate", 0.0))
        f["ch_bsp_zs_cnt"] = _f(feats.get("zs_cnt", 0.0))
        f["ch_bsp_bi_amp"] = _f(feats.get("bsp_bi_amp", 0.0))
        for k in fixed_extra:
            if k in feats:
                f[f"ch_{k}"] = _f(feats[k])
    return f
