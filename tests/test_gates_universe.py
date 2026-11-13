"""宇宙过滤：混合 bars 目录（USDT / sh. / sz. / _MARKET）按市场正确分离。
固化对 1h 目录混放口径的修正（旧 crypto_frenzy_gate.py 未过滤）。
.venv/bin/python tests/test_gates_universe.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from chan_notes.gates import daily_close_wide, universe_filter

# universe_filter 单元断言
assert universe_filter("BTCUSDT", "crypto")
assert not universe_filter("sh.600000", "crypto")
assert not universe_filter("_MARKET", "crypto")
assert universe_filter("sh.600000", "ashare") and universe_filter("sz.000001", "ashare")
assert not universe_filter("BTCUSDT", "ashare")
try:
    universe_filter("X", "nope")
    raise AssertionError("未知市场应抛 ValueError")
except ValueError:
    pass

# 混合目录端到端
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    for code in ["BTCUSDT", "ETHUSDT", "sh.600000", "sz.000001", "_MARKET"]:
        pd.DataFrame({"time": ["2024-01-02 00:00"], "close": [1.0]}
                     ).to_parquet(d / f"{code}.parquet", index=False)
    cw_crypto = daily_close_wide(d, "crypto")
    assert sorted(cw_crypto.columns) == ["BTCUSDT", "ETHUSDT"], cw_crypto.columns
    cw_ashare = daily_close_wide(d, "ashare")
    assert sorted(cw_ashare.columns) == ["sh.600000", "sz.000001"], cw_ashare.columns

print("test_gates_universe: OK")
