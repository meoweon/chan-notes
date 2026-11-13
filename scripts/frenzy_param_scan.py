"""frenzy 门控参数空间扫描（预注册，台账 #47）。

【预注册：跑之前写死，跑完不改】

空间（28 格）
  p_on   ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}   ← IS 目标开机率
  smooth ∈ {1, 3, 5, 10}                          ← frenzy 平滑天数
  冻结：gain_thr=0.08、hold=48、stop=2.0、pos_pct=0.10、max_pos=10、
        切分 2024-01、市场 crypto（30 USDT 1h）、事件集 crypto_10_v1

判读规则（三条同时成立才算"可调"）
  ① 看面不看峰：目标格 (0.5,5) 的 3×3 邻域 OOS Sharpe 极差应 <= 0.3
  ② 可调性：IS 最优格的 OOS 排名应在前 1/3（<= 10/28）
  ③ 山峰真实：OOS 最优格的邻域也要平缓，且格数 >= 3 连成片

试验记账：28 格全部记录，不择优上报；DSR 按有效试验数 28 折减。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from chan_notes.backtest import portfolio
from chan_notes.data import store
from chan_notes.eval.bars import load_frames
from chan_notes.eval.metrics import expectancy
from chan_notes.gates import (daily_close_wide, filter_events,
                              freeze_threshold, compute_frenzy)
from chan_notes.models import train

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "2024-01-01"
P_ON = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SMOOTH = [1, 3, 5, 10]


def run(sub, bars, p):
    res = portfolio.run(sub, bars, p, entry_mode="confirm")
    m = res["metrics"]
    e = expectancy(res["trades"]["ret"])
    return dict(trades=m["n_trades"], total=m["total_ret"] * 100,
                sharpe=m["sharpe"], mdd=m["mdd"] * 100,
                E_bp=e["expectancy_bp"])


def main():
    df = train.load("crypto_10_v1")
    df = df[(df.is_buy == 1) & (df.conf_ok == 1)].copy()
    is_df = df[df["event_time"] < SPLIT]
    oos_df = df[df["event_time"] >= SPLIT]

    close_wide = daily_close_wide(store.BARS / "1h", "crypto")
    frames = load_frames(df.code.unique(), "1h")
    is_bars = {c: b[b["time"] < SPLIT] for c, b in frames.items()}
    oos_bars = {c: b[b["time"] >= SPLIT] for c, b in frames.items()}
    p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=2.0,
                           hold_bars=48, signal_exit=False)
    print(f"事件：IS {len(is_df)} / OOS {len(oos_df)}；门控宇宙 {close_wide.shape[1]} 标的")

    rows = []
    for sm in SMOOTH:
        series = compute_frenzy(close_wide, 0.08, sm)
        for po in P_ON:
            thr = freeze_threshold(series, SPLIT, 1.0 - po)
            a = run(filter_events(is_df, series, thr), is_bars, p)
            b = run(filter_events(oos_df, series, thr), oos_bars, p)
            rows.append(dict(smooth=sm, p_on=po, thr=round(thr, 6),
                             is_trades=a["trades"], is_total=a["total"],
                             is_sharpe=a["sharpe"], is_E=a["E_bp"],
                             oos_trades=b["trades"], oos_total=b["total"],
                             oos_sharpe=b["sharpe"], oos_mdd=b["mdd"],
                             oos_E=b["E_bp"]))
            print(f"  smooth={sm:2d} p_on={po:.1f} thr={thr:.5f} "
                  f"IS sh={a['sharpe']:6.2f} n={a['trades']:4d} | "
                  f"OOS sh={b['sharpe']:6.2f} n={b['trades']:4d} "
                  f"tot={b['total']:+6.1f}%")

    out = pd.DataFrame(rows)
    out["oos_rank"] = out.oos_sharpe.rank(ascending=False).astype(int)
    out["is_rank"] = out.is_sharpe.rank(ascending=False).astype(int)
    path = ROOT / "reports" / "frenzy_param_scan.csv"
    out.to_csv(path, index=False)

    print("\n=== OOS Sharpe 面（行 smooth × 列 p_on） ===")
    print(out.pivot(index="smooth", columns="p_on", values="oos_sharpe")
          .round(2).to_string())
    print("\n=== IS Sharpe 面 ===")
    print(out.pivot(index="smooth", columns="p_on", values="is_sharpe")
          .round(2).to_string())

    base = out[(out.p_on == 0.5) & (out.smooth == 5)].iloc[0]
    nb = out[(out.p_on.isin([0.4, 0.5, 0.6])) & (out.smooth.isin([3, 5, 10]))]
    best_is = out.loc[out.is_sharpe.idxmax()]
    best_oos = out.loc[out.oos_sharpe.idxmax()]
    print(f"\n目标格 (p_on=0.5, smooth=5): OOS Sharpe {base.oos_sharpe:.2f} "
          f"(OOS 排名 {int(base.oos_rank)}/28)")
    print(f"3×3 邻域 OOS Sharpe 区间 [{nb.oos_sharpe.min():.2f}, "
          f"{nb.oos_sharpe.max():.2f}]，极差 {nb.oos_sharpe.max()-nb.oos_sharpe.min():.2f}")
    print(f"IS 最优格 (p_on={best_is.p_on}, smooth={int(best_is.smooth)}): "
          f"IS {best_is.is_sharpe:.2f} → OOS {best_is.oos_sharpe:.2f} "
          f"(OOS 排名 {int(best_is.oos_rank)}/28)")
    print(f"OOS 最优格 (p_on={best_oos.p_on}, smooth={int(best_oos.smooth)}): "
          f"OOS {best_oos.oos_sharpe:.2f} ← IS {best_oos.is_sharpe:.2f} "
          f"(IS 排名 {int(best_oos.is_rank)}/28)")
    print(f"OOS Sharpe 中位 {out.oos_sharpe.median():.2f}，"
          f"为正的格数 {(out.oos_sharpe > 0).sum()}/28")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
