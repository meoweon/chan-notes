"""市场状态门控（regime gate）子包。

统一抽象：日收盘宽表 → 日频指标 → IS 中位数冻结阈值 → 二值事件过滤。
每个门控可单独开关、单独测效果（scripts/gate_lab.py）。
"""
from chan_notes.gates.base import (Gate, binarize, daily_close_wide,
                                   filter_events, freeze_threshold,
                                   is_percentile, universe_filter)
from chan_notes.gates.breadth import BreadthGate, compute_breadth
from chan_notes.gates.frenzy import FrenzyGate, compute_frenzy

# breadth 保留但停用：OOS 夏普 0.17 < always_on 0.41，且在 frenzy 之上是反向减分
# （台账 #35/#36/#47）。需要复核时用 --gates breadth 显式调用。
GATES = {"frenzy": FrenzyGate, "breadth": BreadthGate}

MARKET_DEFAULTS = {
    "crypto": dict(freq="1h", hold=48, stop=2.0, gain_thr=0.08,
                   events="crypto_10_v1"),
    # A股事件集（原 events_v7）与全量 30m 数据缺失，暂缓接入；
    # 参数保留自 frenzy_frozen.py L39（台账 #19 口径）。
    "ashare": dict(freq="30m", hold=144, stop=3.0, gain_thr=0.095, events=None),
}


def get_gate(name: str) -> Gate:
    return GATES[name]()


__all__ = [
    "Gate", "GATES", "MARKET_DEFAULTS", "get_gate",
    "binarize", "daily_close_wide", "filter_events", "freeze_threshold",
    "is_percentile", "universe_filter", "BreadthGate", "compute_breadth",
    "FrenzyGate", "compute_frenzy",
]
