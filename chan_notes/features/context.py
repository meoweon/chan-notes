"""Brooks 个股背景层（context）：信号出现时"市场在哪个阶段、什么位置"。

设计依据（llm-wiki Brooks 概念）：
- fsm-market-cycle：市场周期 spike→channel→TR→barb-wire，区间段应关机
- tight-trading-range：TTR 压倒一切，持续时间比"是不是"更重要
- buy-sell-zones：同一信号在区间顶部/底部赔率完全不同
- consecutive-climaxes：连续高潮计数，第三次后反转概率陡增
- 多周期一致性：30m 信号在日线背景中的位置（只用已完成日线，无泄漏）

全部为因果特征：ind 中的数组第 t 个值只用 bar <= t 的数据。
"""
import numpy as np


def _f(x, default=0.0):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return v


def context_features(ind: dict, t: int) -> dict:
    c = ind["close"][t]
    f = {}

    # 市场周期状态机（Kaufman 效率系数 + 阶段分类）
    f["bg_er20"] = _f(ind["er20"][t])
    f["bg_er60"] = _f(ind["er60"][t])
    f["bg_cycle_stage"] = _f(ind["cycle_stage"][t])
    f["bg_regime_age"] = _f(ind["regime_age"][t])

    # TTR 紧密集约
    f["bg_ttr"] = _f(ind["ttr"][t])
    f["bg_ttr_bars"] = _f(ind["ttr_bars"][t])

    # 买卖区：价格在 50bar 摆动区间的位置（下 1/3 = 买区，上 1/3 = 卖区）
    hh, ll = ind["hh50"][t], ind["ll50"][t]
    if hh == hh and ll == ll and hh > ll and c > 0:
        zone = (c - ll) / (hh - ll)
        f["bg_zone50"] = _f(zone, 0.5)
        f["bg_buy_zone"] = 1.0 if zone <= 1 / 3 else 0.0
        f["bg_sell_zone"] = 1.0 if zone >= 2 / 3 else 0.0
    else:
        f["bg_zone50"] = 0.5
        f["bg_buy_zone"] = 0.0
        f["bg_sell_zone"] = 0.0

    # 连续高潮计数（签名：+n 连续上涨高潮 / -n 下跌）与距上次高潮
    f["bg_climax_cnt"] = _f(ind["climax_cnt"][t])
    f["bg_climax_age"] = _f(ind["climax_age"][t])

    # 日线背景（上一已完成交易日）：MA20 5 日斜率（归一）、60 日区间位置、20 日效率系数
    f["bg_d_ma_slope5"] = _f(ind["d_ma_slope5"][t])
    dh, dl = ind["d_hh60"][t], ind["d_ll60"][t]
    if dh == dh and dl == dl and dh > dl and c > 0:
        f["bg_d_pos60"] = _f((c - dl) / (dh - dl), 0.5)
    else:
        f["bg_d_pos60"] = 0.5
    f["bg_d_er20"] = _f(ind["d_er20"][t])

    return f
