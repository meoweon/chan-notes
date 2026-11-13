"""市场周期状态机（Brooks market cycle 简化）：_MARKET 日线 ER60 + 方向 → trend_up/trend_down/range。
输出 reports/market_regime.csv（d, er60, state），并打印分年分布与历史情节抽查。
严格 PIT：状态在 T 日收盘后可算，供 T+1 日起的事件使用。
"""
import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ER_WIN = 60
ER_THR = 0.25  # Kaufman 常用趋势阈值，冻结

mk = pd.read_parquet(ROOT / "data" / "bars" / "d" / "_MARKET.parquet")
mk["time"] = pd.to_datetime(mk["time"])
mk = mk.set_index("time").sort_index()
c = mk["close"]

net = c - c.shift(ER_WIN)
den = c.diff().abs().rolling(ER_WIN).sum()
er = (net.abs() / den.replace(0, np.nan)).fillna(0.0)
dirn = np.sign(net)

state = pd.Series("range", index=mk.index)
state[(er >= ER_THR) & (dirn > 0)] = "trend_up"
state[(er >= ER_THR) & (dirn < 0)] = "trend_down"

out = pd.DataFrame({"d": mk.index.normalize(), "er60": er.values, "state": state.values})
out.to_csv(ROOT / "reports" / "market_regime.csv", index=False)

# 分年状态分布
out["year"] = pd.to_datetime(out.d).dt.year
tab = out.groupby(["year", "state"]).size().unstack(fill_value=0)
tab["total"] = tab.sum(axis=1)
for s in ("trend_up", "trend_down", "range"):
    if s not in tab.columns:
        tab[s] = 0
print(tab[["trend_up", "trend_down", "range", "total"]].to_string())

# 历史情节抽查（每月末状态）
print("\n情节抽查：")
for d in ("2014-12-31", "2015-06-12", "2016-06-30", "2018-10-19", "2019-04-30",
          "2020-07-10", "2021-12-31", "2022-10-31", "2023-12-29", "2024-06-28",
          "2024-09-30", "2025-06-30", "2026-06-30"):
    row = out[out.d <= d].iloc[-1]
    print(f"  {d}: {row.state:10s} er60={row.er60:.3f}")
print(f"\nsaved {ROOT / 'reports' / 'market_regime.csv'}  er60 分布: "
      f"p25={er.quantile(0.25):.3f} p50={er.quantile(0.5):.3f} p75={er.quantile(0.75):.3f}")
