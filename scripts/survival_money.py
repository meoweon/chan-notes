"""两阶段存活模型的钱验证：预测存活 → 交易期望是否改善。

逻辑：
1. 实际存活 vs 收益：conf_buy 中 survive_144=1/0 两组的 pol144_ret 对比（结构存活本身是否值钱）
2. 预测分数 vs 收益：wf_surv OOF 分数按分位 → 各组 pol144_ret / 存活率（单调性）
3. 组合含义：只做 score top X% 的交易，期望与覆盖率
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ev = pd.read_parquet(ROOT / "data" / "events" / "events_v7s.parquet")
wf = pd.read_parquet(ROOT / "data" / "events" / "wf_surv.parquet")

ok = ev.dropna(subset=["pol144_ret"]).copy()  # conf_buy 且已标记
print(f"conf_buy labeled: n={len(ok)}")

print("\n=== 1. 实际存活 vs 收益（事后视角，不可交易，仅供物理验证）===")
for sv, g in ok.groupby("survive_144"):
    r = g.pol144_ret
    print(f"survive={sv:.0f}: n={len(g)} E={r.mean()*1e4:.1f}bp med={r.median()*1e4:.0f}bp "
          f"win={(r>0).mean()*100:.1f}% tail5%={(r>=r.quantile(0.95)).mean()*100:.1f}%")

print("\n=== 2. WF 存活分数 → 收益（OOF，可交易口径）===")
# wf 是 events_v7s 全量（买卖都有）；只保留 conf_buy 且有 pol144_ret 的
m = wf.merge(ok[["code", "event_time", "pol144_ret"]], on=["code", "event_time"], how="inner")
m = m[m.is_buy == 1].dropna(subset=["pol144_ret"])
print(f"merged OOF conf_buy: n={len(m)}")
m["q"] = pd.qcut(m.score, 5, labels=False)
rows = []
for q, g in m.groupby("q"):
    r = g.pol144_ret
    rows.append({"quintile": f"Q{q+1}", "n": len(g), "score_mean": g.score.mean(),
                 "survive%": g.survive_144.mean() * 100 if "survive_144" in g else np.nan,
                 "E_bp": r.mean() * 1e4, "win%": (r > 0).mean() * 100,
                 "tail_hit%": (r >= ok.pol144_ret.quantile(0.95)).mean() * 100})
print(pd.DataFrame(rows).round(2).to_string(index=False))

print("\n=== 3. 门槛含义 ===")
for thr in (0.5, 0.6, 0.7, 0.8):
    g = m[m.score >= m.score.quantile(thr)]
    r = g.pol144_ret
    print(f"top{(1-thr)*100:.0f}%: n={len(g)} E={r.mean()*1e4:.1f}bp win={(r>0).mean()*100:.1f}% "
          f"vs 全体 E={m.pol144_ret.mean()*1e4:.1f}bp")
