"""gates.base 基础工具对拍：daily_close_wide / freeze_threshold / binarize。
纯 assert 脚本（仓内惯例，无 pytest）：.venv/bin/python tests/test_gates_base.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.gates import binarize, daily_close_wide, freeze_threshold


def _write_bars(dir_: Path, code: str, rows: list[tuple[str, float]]):
    df = pd.DataFrame(rows, columns=["time", "close"])
    df.to_parquet(dir_ / f"{code}.parquet", index=False)


# --- daily_close_wide：日聚合取当日最后一根 bar 收盘 ---
with tempfile.TemporaryDirectory() as td:
    _write_bars(Path(td), "sh.600001", [
        ("2024-01-02 10:00", 10.0), ("2024-01-02 15:00", 11.0),  # 当日最后=11
        ("2024-01-03 15:00", 12.0),
    ])
    _write_bars(Path(td), "sh.600002", [("2024-01-02 15:00", 20.0)])
    w = daily_close_wide(Path(td), "ashare")
    assert list(w.columns) == ["sh.600001", "sh.600002"], w.columns
    assert list(w.index) == ["2024-01-02", "2024-01-03"], w.index
    assert w.loc["2024-01-02", "sh.600001"] == 11.0  # 最后一根，不是第一根
    assert w.loc["2024-01-03", "sh.600001"] == 12.0
    assert pd.isna(w.loc["2024-01-03", "sh.600002"])  # 缺席日为 NaN

# --- freeze_threshold：只用 split 之前样本 ---
s = pd.Series([1.0, 2.0, 3.0, 100.0],
              index=["2023-01-01", "2023-02-01", "2023-03-01", "2024-01-15"])
assert freeze_threshold(s, "2024-01-01") == 2.0  # IS 中位数，不受 2024 极端值影响
try:
    freeze_threshold(s, "2020-01-01")
    raise AssertionError("空 IS 期应抛 ValueError")
except ValueError:
    pass

# --- binarize：>= thr 计入；NaN 关闭 ---
b = binarize(pd.Series([0.4, 0.5, np.nan]), 0.5)
assert list(b) == [False, True, False]

print("test_gates_base: OK")
