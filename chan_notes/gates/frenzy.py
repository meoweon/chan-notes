"""frenzy 门控：极端涨幅占比（市场投机热度代理）。

口径（scripts/frenzy_gate.py L23-38）：每日日涨幅 > gain_thr 的标的占比，
再做 smooth 日平滑。gain_thr：A股 0.095（贴近涨停）/ 加密 0.08
（无涨跌停，早期门控脚本 L21，已归档）。
"""
import pandas as pd

DEFAULT_GAIN_THR = {"ashare": 0.095, "crypto": 0.08}


def compute_frenzy(close_wide: pd.DataFrame, gain_thr: float,
                   smooth: int = 5) -> pd.Series:
    """日涨幅 > gain_thr 的占比，rolling(smooth, min_periods=1) 平滑。"""
    rets = close_wide.pct_change()
    frenzy = (rets > gain_thr).sum(axis=1) / rets.notna().sum(axis=1)
    return frenzy.rolling(smooth, min_periods=1).mean().rename("frenzy")


class FrenzyGate:
    name = "frenzy"

    def __init__(self, gain_thr: float | None = None, smooth: int = 5):
        self.gain_thr = gain_thr
        self.smooth = smooth

    def compute(self, close_wide: pd.DataFrame, market: str) -> pd.Series:
        thr = self.gain_thr if self.gain_thr is not None else DEFAULT_GAIN_THR[market]
        return compute_frenzy(close_wide, thr, self.smooth)
