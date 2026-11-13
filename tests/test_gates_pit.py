"""门控 PIT 检验：事件只能看到前一日的门控值（shift(1)），当日值不泄漏。
.venv/bin/python tests/test_gates_pit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from chan_notes.gates import filter_events

# day2 出现极端高值；thr=10 → 只有 day2 门控为开
s = pd.Series([0.0, 100.0, 0.0], index=["2024-01-01", "2024-01-02", "2024-01-03"])
ev = pd.DataFrame({
    "event_time": ["2024-01-01 10:00", "2024-01-02 10:00", "2024-01-03 10:00"],
    "code": ["X", "X", "X"],
})

kept = filter_events(ev, s, thr=10.0)
# day1 事件：前一日不存在 → 剔除；day2 事件：看到 day1=0（关）→ 剔除；
# day3 事件：看到 day2=100（开）→ 保留。极端值当日的事件吃不到自己的值。
assert list(kept["event_time"]) == ["2024-01-03 10:00"], kept

# 门控值为 NaN 的日前一日，事件被剔除
s_nan = pd.Series([None, 100.0, 0.0], index=s.index)
kept2 = filter_events(ev, s_nan, thr=10.0)
assert list(kept2["event_time"]) == ["2024-01-03 10:00"]

# 全开序列：除首日（无前一日）外全部保留
s_on = pd.Series([100.0, 100.0, 100.0], index=s.index)
assert len(filter_events(ev, s_on, thr=10.0)) == 2

print("test_gates_pit: OK")
