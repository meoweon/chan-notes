"""前缀不变性测试：截断到 T 根 bar 重跑，T 之前发现的事件特征必须与全量一致。

任何差异都意味着特征用到了未来数据（或结构状态被后续数据改写）。
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.core import replay
from chan_notes.data import store
from DataAPI.LocalStore import put_bars


def run(code, df):
    put_bars(code, "30m", df)
    ev, _ = replay.replay(code, "30m", warmup=300, df=df)
    return {(e.bsp_bi_idx, e.bsp_type, e.bar_idx): e for e in ev}


def main(code="sh.600000", cuts=(3000, 6000, 9000)):
    df = store.load(code, "30m")
    full = run(code, df)
    print(f"{code} bars={len(df)} full_events={len(full)}")
    bad = 0
    for T in cuts:
        part = run(code, df.iloc[:T].reset_index(drop=True))
        same, diff = 0, []
        for k, e in part.items():
            if k not in full:
                continue
            f1, f2 = e.features, full[k].features
            d = [n for n in f1 if abs(float(f1[n]) - float(f2.get(n, np.nan))) > 1e-9]
            if d:
                diff.append((k, d[:5]))
            else:
                same += 1
        bad += len(diff)
        print(f"  cut={T}: events={len(part)} matched={same} mismatch={len(diff)}")
        for k, d in diff[:3]:
            print("    ", k, d)
    print("VERDICT:", "PASS" if bad == 0 else f"FAIL ({bad} mismatches)")


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["sh.600000"]))
