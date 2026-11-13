"""行情数组加载：回测与评估共用，避免每个脚本各拼一份 dict。"""
import numpy as np

from chan_notes.data import store


def load_arrays(code: str, freq: str = "30m") -> dict:
    df = store.load(code, freq)
    return {
        "time": df["time"].to_numpy(),
        "open": df["open"].to_numpy(dtype=float),
        "high": df["high"].to_numpy(dtype=float),
        "low": df["low"].to_numpy(dtype=float),
        "close": df["close"].to_numpy(dtype=float),
        "volume": df["volume"].to_numpy(dtype=float),
    }


def load_many(codes, freq: str = "30m") -> dict[str, dict]:
    return {c: load_arrays(c, freq) for c in codes if store.has(c, freq)}


def load_frames(codes, freq: str = "30m") -> dict:
    """组合回测需要带时间列的 DataFrame。"""
    return {c: store.load(c, freq) for c in codes if store.has(c, freq)}


def for_policy(arrays: dict) -> dict:
    """只保留退出模拟需要的四个数组。"""
    return {k: arrays[k] for k in ("open", "high", "low", "close")}
