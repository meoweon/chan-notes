"""用 framework 逐 bar 追踪笔级 BSP 的锚点变化。

目的：回答“BSP 出现后会不会被重绘/挪位”。事件表只存首次出现时的锚点
(`bsp_klu_idx`)，不存之后的锚点；这里对每个 `(code, bi_idx)` 记录首次锚点、
最终锚点、锚点变化次数和出现 bar 区间。

输出：`data/events/crypto_21_v1_anchor_py.parquet`
列：code, bi_idx, first_klu, last_klu, n_changes, first_bar, last_bar, n_seen

用法：
    .venv/bin/python scripts/track_bsp_anchor.py
    .venv/bin/python scripts/track_bsp_anchor.py --codes BTCUSDT,ETHUSDT
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from chan_notes.config import chan_config  # 先把 framework/ 加进 sys.path
from Chan import CChan
from Common.CEnum import AUTYPE, KL_TYPE
from chan_notes.data import store


def track(code: str, freq: str = "1h", warmup: int = 200):
    df = store.load(code, freq)
    chan = CChan(
        code=code,
        data_src="custom:LocalStore.CLocalStore",
        lv_list=[KL_TYPE.K_60M],
        config=chan_config(trigger_step=True),
        autype=AUTYPE.QFQ,
    )

    state = {}
    for snap in chan.step_load():
        lv = snap[0]
        t = lv[-1][-1].idx
        if t < warmup:
            continue
        # 只追踪笔级 BSP：与 replay 的 bs_point_lst.bsp_store_flat_dict 同一口径
        current = {b.bi.idx: b for b in lv.bs_point_lst.bsp_store_flat_dict.values()}
        for bi_idx, bsp in current.items():
            klu = bsp.klu.idx
            rec = state.get(bi_idx)
            if rec is None:
                state[bi_idx] = dict(
                    first_klu=klu, last_klu=klu, n_changes=0,
                    first_bar=t, last_bar=t, n_seen=1,
                )
            else:
                if klu != rec["last_klu"]:
                    rec["n_changes"] += 1
                    rec["last_klu"] = klu
                rec["last_bar"] = t
                rec["n_seen"] += 1

    rows = [dict(code=code, bi_idx=bi_idx, **rec) for bi_idx, rec in state.items()]
    return rows, len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default="", help="逗号分隔；默认事件表里的全部 code")
    ap.add_argument("--freq", default="1h")
    ap.add_argument("--out", default="data/events/crypto_21_v1_anchor_py.parquet")
    args = ap.parse_args()

    if args.codes:
        codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    else:
        ev = pd.read_parquet("data/events/crypto_21_v1.parquet", columns=["code"])
        codes = sorted(ev["code"].unique())

    rows = []
    t0 = time.time()
    for i, code in enumerate(codes, 1):
        recs, n_bars = track(code, args.freq)
        rows.extend(recs)
        print(
            f"[{i}/{len(codes)}] {code} bars={n_bars} bsp_bi={len(recs)} "
            f"elapsed={time.time() - t0:.1f}s",
            flush=True,
        )

    out = pd.DataFrame(rows)
    out.to_parquet(args.out, index=False)
    print(f"saved {args.out} rows={len(out)}")


if __name__ == "__main__":
    main()
