"""本地 parquet 数据源适配器（由 scripts/setup_framework.py 复制到 framework/DataAPI/）。

通过 data_src="custom:LocalStore.CLocalStore" 使用；数据目录默认 <repo>/data/bars，
可用环境变量 CHAN_NOTES_BARS 覆盖。
"""
import os

import pandas as pd

from Common.CEnum import DATA_FIELD, KL_TYPE
from Common.CTime import CTime
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi

FREQ = {
    "K_1M": "1m", "K_5M": "5m", "K_15M": "15m", "K_30M": "30m",
    "K_60M": "1h", "K_DAY": "d", "K_WEEK": "week", "K_MONTH": "month",
}
_DEFAULT_BARS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "bars"))
BARS_ROOT = os.environ.get("CHAN_NOTES_BARS", _DEFAULT_BARS)

_CACHE = {}


def put_bars(code, freq, df):
    _CACHE[(code, freq)] = df


def _load_df(code, freq):
    if (code, freq) in _CACHE:
        return _CACHE[(code, freq)]
    path = os.path.join(BARS_ROOT, freq, f"{code}.parquet")
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    return df


class CLocalStore(CCommonStockApi):
    def __init__(self, code, k_type, begin_date, end_date, autype):
        super(CLocalStore, self).__init__(code, k_type, begin_date, end_date, autype)

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass

    def get_kl_data(self):
        freq = FREQ[self.k_type.name]
        df = _load_df(self.code, freq)
        if self.begin_date:
            df = df[df["time"] >= pd.Timestamp(self.begin_date)]
        if self.end_date:
            df = df[df["time"] <= pd.Timestamp(self.end_date)]
        for r in df.itertuples(index=False):
            t = r.time
            yield CKLine_Unit({
                # auto=False：否则 00:00 的 bar 会被当成日级 K 线（ts 变 23:59），
                # 加密 24/7 数据有真实午夜 bar，会触发 "kline time err"
                DATA_FIELD.FIELD_TIME: CTime(t.year, t.month, t.day, t.hour, t.minute, auto=False),
                DATA_FIELD.FIELD_OPEN: float(r.open),
                DATA_FIELD.FIELD_HIGH: float(r.high),
                DATA_FIELD.FIELD_LOW: float(r.low),
                DATA_FIELD.FIELD_CLOSE: float(r.close),
                DATA_FIELD.FIELD_VOLUME: float(r.volume),
            })
