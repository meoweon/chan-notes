"""给 events_v7 追加 BSP 存活标签。

标签定义（bar 级，H=144）：
- clear_bar >= 0：BSP 在该 bar 从 bs_point_lst 消失（被 clear_store_end 清除=失效）
- clear_bar == -1：回放结束仍存活
- survive_144 = 1 当且仅当 clear_bar==-1 或 clear_bar > bar_idx+144
- 删失（存活但剩余 bar 不足 144）：survive_144 = NaN，训练时排除

产出：events_v7.parquet 加 3 列；另存 events_v7s.parquet（删失行剔除，供 WF）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from chan_notes.data import store
from chan_notes.models import train

H = 144
NAME = sys.argv[1] if len(sys.argv) > 1 else "events_v7"
FREQ = sys.argv[2] if len(sys.argv) > 2 else "30m"

df = train.load(NAME)
n_bars = {c: len(store.load(c, FREQ)) for c in df.code.unique()}
df["n_bars"] = df["code"].map(n_bars)

cb = df["clear_bar"].to_numpy()
bi = df["bar_idx"].to_numpy()
nb = df["n_bars"].to_numpy()

survive = np.where(cb >= 0, (cb > bi + H).astype(float), 1.0)
# 删失：clear_bar==-1 且 bar_idx+H 超出数据末端
censored = (cb < 0) & (bi + H >= nb)
survive = survive.astype(float)
survive[censored] = np.nan

df["survive_144"] = survive
df["cleared_ever"] = (cb >= 0).astype(float)
df["lifespan_bars"] = np.where(cb >= 0, cb - bi, nb - bi)

out = Path(__file__).resolve().parents[1] / "data" / "events" / f"{NAME}.parquet"
df.to_parquet(out, index=False)

ok = df.dropna(subset=["survive_144"])
outs = out.with_name(f"{NAME}s.parquet")
ok.to_parquet(outs, index=False)

print(f"rows={len(df)} cleared_ever={df.cleared_ever.mean()*100:.1f}% "
      f"censored={censored.sum()} ({censored.mean()*100:.2f}%)")
print(f"survive_144 base rate={ok.survive_144.mean()*100:.1f}%  (n={len(ok)})")
for side, sd in ((1, "BUY"), (0, "SELL")):
    s = ok[ok.is_buy == side]
    print(f"  {sd}: n={len(s)} survive={s.survive_144.mean()*100:.1f}% "
          f"cleared_ever={s.cleared_ever.mean()*100:.1f}% "
          f"lifespan_med={s.lifespan_bars.median():.0f}")
print(f"saved {out} + {outs}")
