"""breadth 门控：站上 N 日均线的标的占比。

口径移植自 run/crypto-breadth/scripts/backtest.py L172-183
（breadth_pct_above_ma）：MA 用 rolling(window, min_periods=window)；
分母 = 当日 MA 已定义且有数据的标的数（MA 未定义的早期不贡献）。
简化点：mask=close.notna()，无 crypto-breadth 的 ADV/上市天数流动性过滤。
"""
import numpy as np
import pandas as pd


def compute_breadth(close_wide: pd.DataFrame, window: int = 50) -> pd.Series:
    """每日收盘价 > 自身 window 日均线的标的占比。"""
    ma = close_wide.rolling(window=window, min_periods=window).mean()
    mask = close_wide.notna()
    above = (close_wide > ma) & mask
    n_with_ma = (ma.notna() & mask).sum(axis=1).replace(0, np.nan)
    return (above.sum(axis=1) / n_with_ma).rename("breadth")


class BreadthGate:
    name = "breadth"

    def __init__(self, window: int = 50):
        self.window = window

    def compute(self, close_wide: pd.DataFrame, market: str) -> pd.Series:
        return compute_breadth(close_wide, self.window)
