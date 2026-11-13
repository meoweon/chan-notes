"""BTC 5m 缠论线段状态逐 bar 提取（A 主状态，策略"两腿回调"用）。

逐 bar 快照记录最后一个线段：方向/是否确认/高低点（bi 极值）/中枢数。
产出 data/state/btc5m_state.parquet，与 5m bar 按 bar_idx 对齐。
"""
import io
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["CHAN_NOTES_BARS"] = str(ROOT / "data" / "bars_crypto")  # 必须在 import chan_notes 前设置
sys.path.insert(0, str(ROOT))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import pandas as pd

from chan_notes.config import chan_config
from chan_notes.data import store


def main():
    df = store.load("BTCUSDT", "5m")
    n = len(df)
    times = df["time"].dt.strftime("%Y-%m-%d %H:%M").to_numpy()
    print(f"5m bars={n} span {times[0]} ~ {times[-1]}", flush=True)

    from Chan import CChan
    from Common.CEnum import AUTYPE, KL_TYPE

    chan = CChan(code="BTCUSDT", data_src="custom:LocalStore.CLocalStore",
                 lv_list=[KL_TYPE.K_5M], config=chan_config(trigger_step=True),
                 autype=AUTYPE.QFQ)

    seg_dir = np.zeros(n, dtype=np.int8)
    seg_sure = np.zeros(n, dtype=np.int8)
    seg_high = np.full(n, np.nan)
    seg_low = np.full(n, np.nan)
    seg_nzs = np.zeros(n, dtype=np.int16)

    t0 = time.time()
    seen = 0
    for snap in chan.step_load():
        lv = snap[0]
        klu = lv[-1][-1]
        t = klu.idx
        seen += 1
        segs = lv.seg_list.lst
        seg = segs[-1] if segs else None
        if seg is None:
            seg_dir[t] = 0
            seg_sure[t] = 0
            seg_nzs[t] = 0
        else:
            try:
                seg_high[t] = max(b._high() for b in seg.bi_list)
                seg_low[t] = min(b._low() for b in seg.bi_list)
            except Exception:
                pass
            seg_dir[t] = 1 if seg.is_up() else -1
            seg_sure[t] = 1 if seg.is_sure else 0
            seg_nzs[t] = len(seg.zs_lst)
        if seen % 100000 == 0:
            print(f"  bar {seen}/{n} elapsed={time.time() - t0:.0f}s", flush=True)

    out = pd.DataFrame({
        "bar_idx": np.arange(n), "time": times,
        "seg_dir": seg_dir, "seg_sure": seg_sure,
        "seg_high": seg_high, "seg_low": seg_low, "seg_n_zs": seg_nzs,
    })
    outdir = ROOT / "data" / "state"
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(outdir / "btc5m_state.parquet", index=False)

    up = (seg_dir == 1).mean() * 100
    sure = (seg_sure == 1).mean() * 100
    nan = np.isnan(seg_high).mean() * 100
    print(f"saved {outdir / 'btc5m_state.parquet'} rows={len(out)} elapsed={time.time() - t0:.0f}s")
    print(f"seg_up 占比 {up:.1f}%  seg_sure 占比 {sure:.1f}%  high/low 缺失 {nan:.1f}%")


if __name__ == "__main__":
    main()
