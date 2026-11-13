"""CLEAR 入场前证伪过滤（9.3% 被跳过的交易）单独估价。

adaptive_exit.py 发现：结构失效退出整体大幅劣于固定 144，但 CLEAR 变体附带了一个
入场过滤器（clear_bar <= 入场 bar → 结构在确认入场前已被框架证伪）。
这里单独检验：被过滤掉的交易在固定 144 口径下表现如何？
若为负 → 证伪过滤本身是可用的入场增强；若为正 → 无价值。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from chan_notes.eval.metrics import expectancy

ROOT = Path(__file__).resolve().parents[1]
res = pd.read_parquet(ROOT / "data" / "bt" / "adaptive_exit_events.parquet")

skip = res[res.skip_CLEAR == 1]["ret_F144"].dropna()
keep = res[res.skip_CLEAR == 0]["ret_F144"].dropna()

for name, s in (("被证伪过滤剔除（CLEAR skip）", skip), ("保留", keep)):
    ex = expectancy(s)
    print(f"{name}: n={ex['n']} E={ex['expectancy_bp']:+.1f}bp "
          f"win={ex['win_rate']*100:.1f}% R={ex['payoff_R']:.2f} "
          f"median={ex['median_ret_bp']:+.0f}bp")

# 合并检验：F144 + 证伪过滤（不动退出，只加入场过滤）
ex_all = expectancy(res["ret_F144"].dropna())
print(f"无过滤基线: n={ex_all['n']} E={ex_all['expectancy_bp']:+.1f}bp")
