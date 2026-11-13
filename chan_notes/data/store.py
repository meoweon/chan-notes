"""Parquet 缓存管理。

数据按 (code, freq) 存储在 data/bars/{freq}/{code}.parquet。
"""
import json
from pathlib import Path

import pandas as pd

from chan_notes.config import DATA

BARS = DATA / "bars"
UNIVERSE = DATA / "universe"


def _path(code: str, freq: str) -> Path:
    return BARS / freq / f"{code}.parquet"


def save(code: str, df: pd.DataFrame, freq: str):
    """保存 DataFrame 到 parquet。"""
    p = _path(code, freq)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, engine="pyarrow")


def has(code: str, freq: str) -> bool:
    """检查数据是否已缓存。"""
    return _path(code, freq).exists()


def load(code: str, freq: str) -> pd.DataFrame:
    """从 parquet 加载 DataFrame。"""
    p = _path(code, freq)
    if not p.exists():
        raise FileNotFoundError(f"未找到 {code} {freq} 数据：{p}")
    return pd.read_parquet(p, engine="pyarrow")


def list_codes(freq: str) -> list[str]:
    """列出指定频率下所有已缓存的代码。"""
    d = BARS / freq
    if not d.exists():
        return []
    return [p.stem for p in d.glob("*.parquet")]
