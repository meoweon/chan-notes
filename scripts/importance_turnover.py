"""特征重要性轮换分析：每个训练窗口，模型到底在用哪些特征？

回答"这一阶段哪些特征有效、下一窗口是否换了一批"：
- top-20 特征在相邻窗口的重合率（Jaccard）；
- 全量 gain 向量的秩相关（Spearman）；
- 每个特征进入 top-20 的月份占比（稳定性榜）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main(name="wf_win12", topk=20):
    imp = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "events" / f"{name}_importance.parquet")
    months = sorted(imp.month.unique())
    tops, vecs = {}, {}
    for m, g in imp.groupby("month"):
        g = g.sort_values("gain", ascending=False)
        tops[m] = set(g.head(topk).feature)
        vecs[m] = g.set_index("feature")["gain"]

    jac, rho = [], []
    for a, b in zip(months, months[1:]):
        jac.append(len(tops[a] & tops[b]) / len(tops[a] | tops[b]))
        al = vecs[a].to_frame("ga").join(vecs[b].to_frame("gb"), how="inner")
        rho.append(spearmanr(al.ga, al.gb).statistic if len(al) > 5 else np.nan)
    print(f"{name}: {len(months)} 个窗口")
    print(f"相邻窗口 top{topk} 重合率: mean={np.mean(jac):.2f}  min={np.min(jac):.2f}")
    print(f"相邻窗口 gain 秩相关:   mean={np.nanmean(rho):.2f}  min={np.nanmin(rho):.2f}")

    freq = {}
    for m, s in tops.items():
        for f in s:
            freq[f] = freq.get(f, 0) + 1
    stab = pd.Series(freq).sort_values(ascending=False) / len(months)
    print(f"\n稳定性榜（进入 top{topk} 的月份占比）：")
    print(stab.head(25).round(2).to_string())
    print(f"\n最不稳定（进过 top{topk} 但占比<20%）：{(stab < 0.2).sum()} 个")
    out = stab.rename("month_share").rename_axis("feature").to_frame()
    out.to_csv(Path(__file__).resolve().parents[1] / "reports" / f"importance_stability_{name}.csv")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["wf_win12"]))
