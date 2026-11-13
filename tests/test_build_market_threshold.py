"""指数在场数阈值：绝对值不得静默抹掉早期历史，筛空/超篮子必须硬失败。
固化 D038 对 min_stocks=30（A 股时代绝对值）的修正，并冻结旧口径不漂移。
全部用合成篮子 + 临时目录，不读写真实 data/bars。
.venv/bin/python tests/test_build_market_threshold.py
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.data import store

_spec = importlib.util.spec_from_file_location(
    "build_market", Path(__file__).resolve().parents[1] / "scripts" / "build_market.py")
_bm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bm)
build_market = _bm.build_market

FREQ = "1h"
T0 = pd.Timestamp("2020-01-01")


def _basket(n: int, stagger: int = 0, bars: int = 120) -> list[str]:
    """造 n 个标的；stagger>0 时第 i 个晚 i*stagger 小时才上市，模拟篮子随时间增长。"""
    codes = []
    for i in range(n):
        times = pd.date_range(T0 + pd.Timedelta(hours=i * stagger), periods=bars, freq="h")
        idx = np.arange(i * stagger, i * stagger + bars)
        df = pd.DataFrame({
            "time": times,
            "close": 100.0 * np.exp(0.001 * idx + 0.05 * np.sin(idx / 7 + i)),
        })
        code = f"T{i:03d}USDT"
        store.save(code, df, FREQ)
        codes.append(code)
    return codes


def _legacy(codes: list[str], min_stocks: int) -> pd.Series:
    """修正前的原公式，用于证明默认口径逐值不变。"""
    px = pd.concat(
        [store.load(c, FREQ)[["time", "close"]].rename(columns={"close": c}).set_index("time")
         for c in codes], axis=1).sort_index()
    px = px.ffill(limit=8)
    level = (px / px.bfill().iloc[0]).median(axis=1)
    level = level[px.notna().sum(axis=1) >= min_stocks]
    return 1000 * (1 + level.pct_change().fillna(0)).cumprod()


with tempfile.TemporaryDirectory() as td:
    store.BARS = Path(td) / "bars"

    # 1. 阈值大于篮子：硬失败，且报错要点名篮子大小（旧版是 IndexError）
    small = _basket(3)
    try:
        build_market(FREQ, min_stocks=30, codes=small, dry_run=True)
        raise AssertionError("阈值超篮子应抛 ValueError")
    except ValueError as e:
        assert "大于篮子大小 3" in str(e), str(e)
    assert not store.has("_MARKET", FREQ), "失败路径不得落盘"

    # 2. 阈值筛空：硬失败，不得写空指数
    sparse = _basket(40, stagger=200, bars=1)
    try:
        build_market(FREQ, min_stocks=30, codes=sparse, dry_run=True)
        raise AssertionError("筛空应抛 ValueError")
    except ValueError as e:
        assert "拒绝写空指数" in str(e), str(e)
    assert not store.has("_MARKET", FREQ)

    # 3. 增长篮子：绝对阈值抹掉早期，相对阈值保住早期
    grow = _basket(40, stagger=1)
    abs_df = build_market(FREQ, min_stocks=30, codes=grow, dry_run=True)
    rel_df = build_market(FREQ, min_frac=0.5, codes=grow, dry_run=True)
    assert abs_df["time"].min() == T0 + pd.Timedelta(hours=29), abs_df["time"].min()
    assert rel_df["time"].min() == T0, rel_df["time"].min()
    # 并集 0~158 共 159 根：绝对阈值砍头 29 + 砍尾 21；相对阈值只砍尾 11（成员退市）
    assert len(abs_df) == 109, len(abs_df)
    assert len(rel_df) == 148, len(rel_df)

    # 4. 默认口径逐值不变（与修正前原公式对撞）
    ref = _legacy(grow, 30)
    got = abs_df.set_index("time")["close"]
    assert len(ref) == len(got)
    assert np.allclose(ref.values, got.values, rtol=0, atol=1e-9), float((ref - got).abs().max())

    # 5. dry-run 不落盘；非 dry-run 才写
    assert not store.has("_MARKET", FREQ)
    build_market(FREQ, min_stocks=30, codes=grow)
    assert store.has("_MARKET", FREQ)
    assert len(store.load("_MARKET", FREQ)) == len(abs_df)

    # 6. min_frac 越界硬失败
    for bad in (0.0, -0.1, 1.5):
        try:
            build_market(FREQ, min_frac=bad, codes=grow, dry_run=True)
            raise AssertionError(f"min_frac={bad} 应抛 ValueError")
        except ValueError:
            pass

    # 7. 合成序列自身的不变量（缠论引擎校验前提）
    out = store.load("_MARKET", FREQ)
    assert out["close"].iloc[0] == 1000.0
    assert (out["volume"] == 0).all()
    assert (out["high"] >= out[["open", "close"]].max(axis=1)).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1)).all()
    assert out["time"].is_monotonic_increasing
    assert not out[["open", "high", "low", "close"]].isna().any().any()

    # 8. 空篮子硬失败
    try:
        build_market("4h", dry_run=True)
        raise AssertionError("空篮子应抛 ValueError")
    except ValueError as e:
        assert "篮子为空" in str(e), str(e)

print("test_build_market_threshold: 8 组断言全过")