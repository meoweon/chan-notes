"""市场环境特征：用沪深300 30分钟线的因果状态做背景，按时间对齐到个股事件。"""
import numpy as np
import pandas as pd


class MarketContext:
    def __init__(self, idx_df: pd.DataFrame):
        from chan_notes.features import indicators
        self.ind = indicators.precompute(idx_df)
        self.times = pd.to_datetime(idx_df["time"]).to_numpy()

    def features_at(self, ts) -> dict:
        t = np.searchsorted(self.times, np.datetime64(pd.Timestamp(ts)), side="right") - 1
        if t < 60:
            return {k: 0.0 for k in self._keys()}
        c = self.ind["close"][t]
        ma20, ma60 = self.ind["ma20"][t], self.ind["ma60"][t]
        ret20 = c / self.ind["close"][t - 20] - 1 if t >= 20 else 0.0
        ret5 = c / self.ind["close"][t - 5] - 1 if t >= 5 else 0.0
        atr = self.ind["atr14"][t]
        return {
            "mkt_ret5": float(ret5),
            "mkt_ret20": float(ret20),
            "mkt_above_ma20": 1.0 if (ma20 == ma20 and c > ma20) else 0.0,
            "mkt_above_ma60": 1.0 if (ma60 == ma60 and c > ma60) else 0.0,
            "mkt_ma20_slope": float(ma20 / self.ind["ma20"][t - 20] - 1) if t >= 20 and self.ind["ma20"][t - 20] == self.ind["ma20"][t - 20] and self.ind["ma20"][t - 20] > 0 else 0.0,
            "mkt_atr_pct": float(atr / c) if atr == atr and c > 0 else 0.0,
            "mkt_rsi": float(self.ind["rsi14"][t]) if self.ind["rsi14"][t] == self.ind["rsi14"][t] else 50.0,
            "mkt_range_pos20": float((c - self.ind["ll20"][t]) / (self.ind["hh20"][t] - self.ind["ll20"][t]))
            if self.ind["hh20"][t] == self.ind["hh20"][t] and self.ind["hh20"][t] > self.ind["ll20"][t] else 0.5,
        }

    @staticmethod
    def _keys():
        return ("mkt_ret5", "mkt_ret20", "mkt_above_ma20", "mkt_above_ma60",
                "mkt_ma20_slope", "mkt_atr_pct", "mkt_rsi", "mkt_range_pos20")
