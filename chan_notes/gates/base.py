"""门控（regime gate）公共协议与工具。

统一抽象：日收盘宽表 → 日频横截面指标 → IS 中位数冻结阈值 → 二值事件过滤。
PIT 纪律：事件日映射前一日的门控值（series.shift(1)），
对齐 scripts/frenzy_frozen.py L24（台账 #19 范式）。
"""
from pathlib import Path
from typing import Protocol

import pandas as pd


class Gate(Protocol):
    """日频横截面门控：输入日收盘宽表，输出日频原始指标序列（index 升序）。"""

    name: str

    def compute(self, close_wide: pd.DataFrame, market: str) -> pd.Series: ...


def universe_filter(stem: str, market: str) -> bool:
    """按市场前缀过滤 bars 文件名（_MARKET 恒排除）。

    口径修正：data/bars/1h/ 混放 USDT 与 A股 parquet，旧
    早期门控脚本未过滤，把 A股混进了加密狂热指数。
    """
    if stem == "_MARKET":
        return False
    if market == "crypto":
        return stem.endswith("USDT")
    if market == "ashare":
        return stem.startswith(("sh.", "sz."))
    raise ValueError(f"未知市场: {market}")


def daily_close_wide(bar_dir: Path, market: str) -> pd.DataFrame:
    """bars 目录 → 日收盘宽表（index=YYYY-MM-DD 字符串升序，columns=code）。

    日聚合口径：bar time 取前 10 字符（同 scripts/frenzy_gate.py L29），
    A股=交易日、加密=UTC 自然日，各自自洽；shift(1) 语义随之分别为
    前一交易日 / 前一自然日，跨市场不可直接比阈值。
    """
    daily = {}
    for f in sorted(bar_dir.glob("*.parquet")):
        if not universe_filter(f.stem, market):
            continue
        b = pd.read_parquet(f, columns=["time", "close"])
        b["d"] = b["time"].astype(str).str[:10]
        daily[f.stem] = b.groupby("d")["close"].last()
    if not daily:
        raise ValueError(f"{bar_dir} 下没有 market={market} 的标的")
    return pd.DataFrame(daily).sort_index()


def freeze_threshold(series: pd.Series, split: str, q: float = 0.5) -> float:
    """IS 分位冻结阈值：只用 split 之前的样本估计（frenzy_frozen.py L25 纪律）。

    q 由"开机率"反推：q = 1 - p_on。默认 0.5 即中位数，与旧口径逐位一致。
    注意这是**一次性冻结的常数**，不是滚动分位——滚动分位度量相对热，
    已在台账 #19 对照中判死（夏普 0.21）。
    """
    is_part = series[series.index < split].dropna()
    if is_part.empty:
        raise ValueError(f"IS 期（< {split}）没有有效门控值")
    return float(is_part.quantile(q))


def is_percentile(series: pd.Series, split: str, value: float) -> float:
    """value 落在 IS 分布的第几分位（0-100），把阈值翻译成人话。"""
    is_part = series[series.index < split].dropna()
    if is_part.empty:
        raise ValueError(f"IS 期（< {split}）没有有效门控值")
    return float((is_part <= value).mean() * 100.0)


def binarize(series: pd.Series, thr: float) -> pd.Series:
    """``>= thr`` 为开；NaN 一律关闭。"""
    return (series >= thr).where(series.notna(), False)


def filter_events(events: pd.DataFrame, series: pd.Series, thr: float,
                  time_col: str = "event_time") -> pd.DataFrame:
    """二值门控事件过滤：事件日映射前一日的门控开关（PIT，shift(1)）。

    对齐 scripts/frenzy_frozen.py L24-33 范式；门控值缺失日的事件被剔除。
    """
    prev = binarize(series, thr).shift(1)
    d = pd.to_datetime(events[time_col]).dt.strftime("%Y-%m-%d")
    on = d.map(prev)
    return events[on.fillna(False)].copy()
