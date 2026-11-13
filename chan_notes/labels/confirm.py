"""确认入场：信号 bar 之后 N 根内，收盘突破信号 bar 高/低点才入场（Brooks 的二次确认）。"""
import numpy as np


def confirmation_entry(ind: dict, t: int, is_buy: int, max_wait: int = 4):
    """返回 (ok, entry_idx, wait_bars)。entry_idx 为确认 bar 的下一根。"""
    n = len(ind["close"])
    ref = ind["high"][t] if is_buy else ind["low"][t]
    for i in range(t + 1, min(t + 1 + max_wait, n - 1)):
        if is_buy and ind["close"][i] > ref:
            return True, i + 1, i - t
        if (not is_buy) and ind["close"][i] < ref:
            return True, i + 1, i - t
    return False, -1, -1
