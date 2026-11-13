"""frenzy 冻结门控的稳健性检查：分年度拆解 + 去尾测试。

读 data/bt/frozen_{always_on,frozen_gate}_trades.parquet（frenzy_frozen.py 产出）。
问题：门控版的 +79.5% 是否全靠 2024-09 一个月？去掉尾部后还剩什么？
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

rows_year, rows_tail = [], []
for tag in ("always_on", "frozen_gate"):
    tr = pd.read_parquet(ROOT / "data" / "bt" / f"frozen_{tag}_trades.parquet")
    tr["year"] = pd.to_datetime(tr["entry_time"]).dt.year
    tr["ym"] = pd.to_datetime(tr["entry_time"]).dt.to_period("M").astype(str)
    for y, g in tr.groupby("year"):
        rows_year.append({
            "variant": tag, "year": y, "n": len(g),
            "E_bp": g.ret.mean() * 1e4, "win%": (g.ret > 0).mean() * 100,
            "sum_bp": g.ret.sum() * 1e4,
        })
    # 最热月份
    hot = tr.groupby("ym").ret.agg(["count", "mean"]).sort_values("mean", ascending=False).head(3)
    print(f"\n[{tag}] 最热月份:")
    print((hot.assign(mean=hot["mean"] * 1e4).round(0)).to_string())
    # 去尾
    n = len(tr)
    for cut in (0.0, 0.01, 0.05, 0.10):
        k = int(np.floor(n * cut))
        keep = tr.sort_values("ret").iloc[: n - k] if k else tr
        rows_tail.append({
            "variant": tag, "cut": f"top{cut:.0%}" if cut else "none",
            "n": len(keep), "E_bp": keep.ret.mean() * 1e4,
            "win%": (keep.ret > 0).mean() * 100,
        })

print("\n=== 分年度 ===")
print(pd.DataFrame(rows_year).round(1).to_string(index=False))
print("\n=== 去尾 ===")
print(pd.DataFrame(rows_tail).round(1).to_string(index=False))

pd.DataFrame(rows_year).to_csv(ROOT / "reports" / "frenzy_frozen_yearly.csv", index=False)
pd.DataFrame(rows_tail).to_csv(ROOT / "reports" / "frenzy_frozen_tail.csv", index=False)
print("saved frenzy_frozen_yearly.csv / frenzy_frozen_tail.csv")
