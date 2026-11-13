"""非狂潮期用模型筛选是否可行 —— 第一步：事件级（预注册，台账 #50）。

【预注册：跑之前写死，跑完不改】

假设
  H：frenzy 关机期整体 E 为负（前测 −13.5bp），但模型分数在关机期内仍能分层，
     即"关机 + 高存活分"的事件 E 显著为正，足以覆盖 20bp 往返成本。
  这是"互斥分工"（狂潮全放开、冷淡用模型）的第一道闸。本步只做事件级，不跑组合。

数据与口径
  事件集：crypto_10_v1（主，与策略同口径）+ crypto_31_v1（副，为增加样本量）
  过滤：is_buy==1 & conf_ok==1 & event_time >= 2024-01-01（真 OOS）
  门控：frenzy5 @ p_on=0.5（IS 冻结阈值），PIT shift(1)
  分数：wf_crypto_31_xgb_oos 的 walk-forward OOS 概率（oos_prob），
        按月滚动训练、当月样本外，非全样本拟合
  出场：固定 48 bar / 2×ATR，无仓位约束（max_pos 放开），与前测同规则

判据（两条同时成立才进入组合级）
  ① 关机子集最高五分位 E > 30bp
  ② 关机子集五分位 E 单调递增（Spearman(分位序号, E) >= 0.8）

附带对照
  开机子集做同样五分位，预期不单调或无增益（验证"狂潮期不该筛"，与 #22/#33 一致）

解读纪律
  样本小，分位 E 的标准误约 50–100bp；即使点估计达线，也要看 95% CI 是否排除 0。
  本步阴性则直接判死，不再进入组合级，也不调分位切点。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pickle

from chan_notes.backtest import portfolio
from chan_notes.data import store
from chan_notes.eval.bars import load_frames
from chan_notes.gates import daily_close_wide, freeze_threshold, compute_frenzy

ROOT = Path(__file__).resolve().parents[1]
SPLIT = "2024-01-01"
MODEL = "wf_crypto_31_xgb_oos"


def scores() -> pd.DataFrame:
    o = pickle.load(open(ROOT / f"data/models/{MODEL}.pkl", "rb"))
    src = pd.read_parquet(ROOT / "data/events/crypto_31_v1.parquet").reset_index(drop=True)
    src["score"] = np.nan
    src.loc[np.asarray(o["oos_idx"]), "score"] = np.asarray(o["oos_prob"])
    return src[["code", "event_time", "is_buy", "bsp_type", "score"]].drop_duplicates(
        ["code", "event_time", "is_buy", "bsp_type"])


def prep(events_name: str, sc: pd.DataFrame):
    d = pd.read_parquet(ROOT / f"data/events/{events_name}.parquet")
    d = d[(d.is_buy == 1) & (d.conf_ok == 1)].copy()
    d = d.merge(sc, on=["code", "event_time", "is_buy", "bsp_type"], how="left")
    d["entry_time"] = pd.to_datetime(d["entry_time"])
    d["conf_entry_time"] = pd.to_datetime(d["conf_entry_time"])
    d = d[d.event_time >= SPLIT]
    return d[d.score.notna()].copy()


def main():
    sc = scores()
    series = compute_frenzy(daily_close_wide(store.BARS / "1h", "crypto"), 0.08, 5)
    thr = freeze_threshold(series, SPLIT, 0.5)
    p = portfolio.BtParams(pos_pct=0.10, max_pos=10_000, stop_atr=2.0,
                           hold_bars=48, signal_exit=False)
    print(f"frenzy 阈值 {thr:.5f}（p_on=0.5，IS 冻结）\n")

    for name in ["crypto_10_v1", "crypto_31_v1"]:
        d = prep(name, sc)
        frames = load_frames(d.code.unique(), "1h")
        bars = {c: b[b["time"] >= SPLIT] for c, b in frames.items()}
        res = portfolio.run(d, bars, p, entry_mode="confirm")
        t = res["trades"][["code", "entry_time", "ret"]].copy()
        m = t.merge(d[["code", "conf_entry_time", "score"]], left_on=["code", "entry_time"],
                    right_on=["code", "conf_entry_time"], how="left")
        m = m[m.score.notna()].copy()
        lag = series.shift(1)
        lag.index = pd.DatetimeIndex(lag.index).normalize()
        m["gate"] = m.entry_time.dt.normalize().map(lag).fillna(-1.0) >= thr

        print(f"=== {name}：{len(m)} 笔可评估（开机 {int(m.gate.sum())}）===")
        for g, sub in [("关机", m[~m.gate]), ("开机", m[m.gate])]:
            if len(sub) < 25:
                print(f"  {g}: n={len(sub)} 过少，跳过")
                continue
            sub = sub.copy()
            sub["q"] = pd.qcut(sub.score, 5, labels=[1, 2, 3, 4, 5])
            rows = []
            for q, gg in sub.groupby("q", observed=True):
                r = gg.ret.values
                se = r.std(ddof=1) / np.sqrt(len(r)) * 1e4
                rows.append([int(q), len(r), r.mean() * 1e4, se, (r > 0).mean() * 100])
            print(f"  {g}子集（n={len(sub)}）")
            print("    分位   n     E(bp)   SE   胜率%")
            for q, n, e, se, w in rows:
                print(f"      {q}   {n:5d}  {e:+8.1f}  {se:5.1f}  {w:5.1f}")
            es = [x[2] for x in rows]
            rho = pd.Series(es).corr(pd.Series(range(1, len(es) + 1)), method="spearman")
            top_ok = es[-1] > 30
            print(f"    最高分位 {es[-1]:+.1f}bp（>30? {top_ok}）；单调 Spearman={rho:.2f}（>=0.8? {rho >= 0.8}）")
            if g == "关机":
                print(f"    → 判据 {'通过，可进组合级' if (top_ok and rho >= 0.8) else '不通过，判死'}")
            print()


if __name__ == "__main__":
    main()
