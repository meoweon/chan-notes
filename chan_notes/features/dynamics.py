"""动力学内部度量特征（dy_ 前缀）：MACD 多算法笔动量、中枢进出笔、量能、特征序列。

与 chan_ctx.py 的分工：
- chan_ctx 收结构几何（幅度/斜率/位置/中枢区间）+ BSP 原生 features（单一 macd_algo 的 divergence_rate）
- 本模块收官方 ZS.is_divergence 语义下 6 种 MACD 算法的进/出笔动量及比率、笔级量能、
  中枢进出笔专项比率、eigen 特征序列状态

因果性：全部取 BSP 首次出现 bar 上已完成的结构（out_bi 已结束、zs/seg 为当时可见状态），无前视。
口径：in 笔 is_reverse=False、out 笔 is_reverse=True，与 framework/ZS/ZS.py is_divergence 一致。
"""
from Common.CEnum import MACD_ALGO


def _f(x, d=0.0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return d
    if v != v or v in (float("inf"), float("-inf")):
        return d
    return v


def _ratio(out, inn, d=1.0):
    try:
        v = float(out) / (float(inn) + 1e-7)
    except (TypeError, ValueError, ZeroDivisionError):
        return d
    if v != v or v in (float("inf"), float("-inf")):
        return d
    return v


# (特征名后缀, MACD_ALGO) —— slope/amp 实际与 MACD 无关（幅度/斜率），沿用框架命名
_MACD_ALGOS = [
    ("area", MACD_ALGO.AREA),        # 半区 MACD 柱面积（背驰默认口径之一）
    ("farea", MACD_ALGO.FULL_AREA),  # 全笔同号柱面积
    ("peak", MACD_ALGO.PEAK),        # 柱峰值（本项目 divergence 判定口径）
    ("diff", MACD_ALGO.DIFF),        # 红绿柱极差
    ("slope", MACD_ALGO.SLOPE),      # 笔斜率（归一）
    ("amp", MACD_ALGO.AMP),          # 笔幅度（归一）
]


def _metric(bi, algo, rev):
    if bi is None:
        return None
    try:
        return float(bi.cal_macd_metric(algo, rev))
    except Exception:
        return None


def dynamics_features(lv, bsp) -> dict:
    f = {}
    out_bi = getattr(bsp, "bi", None)
    bis = lv.bi_list
    in_bi = None
    if out_bi is not None and out_bi.idx >= 2:
        cand = bis[out_bi.idx - 2]  # 笔方向交替，idx-2 即前一同向笔
        if cand.is_up() == out_bi.is_up():
            in_bi = cand

    # --- 1) 多算法笔动量（官方背驰语义：in 正序、out 逆序） ---
    for name, algo in _MACD_ALGOS:
        o = _metric(out_bi, algo, True)
        i = _metric(in_bi, algo, False)
        f[f"dy_macd_{name}_out"] = _f(o)
        f[f"dy_macd_{name}_in"] = _f(i)
        f[f"dy_macd_{name}_ratio"] = _ratio(o, i) if (o is not None and i is not None) else 1.0

    # --- 2) RSI（需 config cal_rsi=True） ---
    f["dy_rsi_out"] = _f(_metric(out_bi, MACD_ALGO.RSI, True))
    f["dy_rsi_in"] = _f(_metric(in_bi, MACD_ALGO.RSI, False))

    # --- 3) 笔级量能（总量/均值） ---
    for name, algo in (("vol", MACD_ALGO.VOLUMN), ("volavg", MACD_ALGO.VOLUMN_AVG)):
        o = _metric(out_bi, algo, False)
        i = _metric(in_bi, algo, False)
        f[f"dy_{name}_out"] = _f(o)
        f[f"dy_{name}_in"] = _f(i)
        f[f"dy_{name}_ratio"] = _ratio(o, i) if (o is not None and i is not None) else 1.0

    # --- 4) 中枢进出笔动力学（最后可见中枢） ---
    f["dy_zs_break"] = 0.0
    f["dy_zs_area_ratio"] = 1.0
    f["dy_zs_peak_ratio"] = 1.0
    f["dy_zs_out_is_peak"] = 0.0
    f["dy_zs_peak_rate"] = 0.0
    zss = lv.zs_list.zs_lst
    if zss and out_bi is not None:
        zs = zss[-1]
        in_zs = getattr(zs, "bi_in", None)
        try:
            f["dy_zs_break"] = 1.0 if zs.end_bi_break(out_bi) else 0.0
        except Exception:
            pass
        if in_zs is not None:
            o = _metric(out_bi, MACD_ALGO.AREA, True)
            i = _metric(in_zs, MACD_ALGO.AREA, False)
            if o is not None and i is not None:
                f["dy_zs_area_ratio"] = _ratio(o, i)
            o = _metric(out_bi, MACD_ALGO.PEAK, True)
            i = _metric(in_zs, MACD_ALGO.PEAK, False)
            if o is not None and i is not None:
                f["dy_zs_peak_ratio"] = _ratio(o, i)
        try:
            is_peak, rate = zs.out_bi_is_peak(out_bi.idx)
            f["dy_zs_out_is_peak"] = 1.0 if is_peak else 0.0
            if rate is not None:
                f["dy_zs_peak_rate"] = _f(rate)
        except Exception:
            pass

    # --- 5) 特征序列（最近一次由 eigen 确认的线段） ---
    f["dy_seg_eigen_has"] = 0.0
    f["dy_seg_eigen_gap"] = 0.0
    f["dy_seg_eigen_dist"] = 0.0
    if out_bi is not None:
        for seg in reversed(lv.seg_list.lst):
            efx = getattr(seg, "eigen_fx", None)
            if efx is None or efx.ele[1] is None:
                continue
            f["dy_seg_eigen_has"] = 1.0
            f["dy_seg_eigen_gap"] = 1.0 if efx.ele[1].gap else 0.0
            try:
                f["dy_seg_eigen_dist"] = float(max(out_bi.idx - seg.bi_list[-1].idx, 0))
            except Exception:
                pass
            break
    return f
