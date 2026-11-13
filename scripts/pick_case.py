"""为存活模型找一对直观案例：低分且被清除（事后大跌）vs 高分且存活（事后大涨）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from chan_notes.models import train

ROOT = Path(__file__).resolve().parents[1]

wf = pd.read_parquet(ROOT / "data" / "events" / "wf_surv.parquet")
print("wf_surv columns:", wf.columns.tolist()[:20])
print(wf.head(2).to_string())
print("rows:", len(wf))
