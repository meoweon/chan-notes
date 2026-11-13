"""frenzy / breadth 手算对拍（含首日 NaN、rolling min_periods、MA 未定义分母剔除）。
.venv/bin/python tests/test_gates_values.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.gates import compute_breadth, compute_frenzy

# --- frenzy：3 代码 4 日，gain_thr=0.08，smooth=2 ---
# rets: d1 全 NaN（pct_change 首行）；d2: A +9%（计）, B 0%, C +1%
#       d3: A 0%, B +10%（计）, C -1%；d4: A +9%（计）, B +9%（计）, C 0%
w = pd.DataFrame({
    "A": [100.0, 109.0, 109.0, 118.81],
    "B": [100.0, 100.0, 110.0, 119.9],
    "C": [100.0, 101.0, 100.0, 100.0],
}, index=["d1", "d2", "d3", "d4"])
fz = compute_frenzy(w, gain_thr=0.08, smooth=2)
# 原始 frenzy: d1=NaN(0/0), d2=1/3, d3=1/3, d4=2/3
# rolling(2, min_periods=1).mean() 跳过 NaN：d2=(1/3), d3=(1/3+1/3)/2, d4=(1/3+2/3)/2
assert pd.isna(fz["d1"])
assert abs(fz["d2"] - 1 / 3) < 1e-12
assert abs(fz["d3"] - (1 / 3 + 1 / 3) / 2) < 1e-12
assert abs(fz["d4"] - (1 / 3 + 2 / 3) / 2) < 1e-12

# --- breadth：window=3，C 前 3 日缺席（MA 未定义 → 不进分母） ---
w2 = pd.DataFrame({
    "A": [100.0, 90.0, 80.0, 95.0, 100.0, 110.0],
    "B": [100.0, 110.0, 105.0, 95.0, 90.0, 85.0],
    "C": [np.nan, np.nan, np.nan, 100.0, 105.0, 110.0],
}, index=["d1", "d2", "d3", "d4", "d5", "d6"])
br = compute_breadth(w2, window=3)
assert pd.isna(br["d1"]) and pd.isna(br["d2"])          # MA 未定义期整体 NaN
assert br["d3"] == 0.0          # A 80<MA90, B 105==MA105 不算站上；C 不进分母
assert br["d4"] == 0.5          # A 95>88.33, B 95<103.33；分母=2
assert br["d5"] == 0.5          # A 100>91.67, B 90<96.67；C 仍不足 3  obs
assert abs(br["d6"] - 2 / 3) < 1e-12  # C MA=105<110 计入；A上B下 → 2/3

print("test_gates_values: OK")
