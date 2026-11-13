"""E002 Layer 1：对 480 组参数扫描结果做统计稳健性检验。

三个部分：
  A. 秩相关（IS↔OOS 夏普的 Spearman + 置换 p 值）与网格分布，只读 CSV。
  B. 块自举（按月分块）给 基线 / IS 最优 / OOS 最优 三组的 OOS 夏普、收益算 95% CI。
  C. 置换检验：IS 最优组合的门控 vs「等量事件随机抽样」，构造零分布。

用法：
  python scripts/analyze_sweep_significance.py                 # A+B
  python scripts/analyze_sweep_significance.py --skip-b --perms 500   # 只跑 C
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

import param_sweep_backtest as psb
from chan_notes.backtest import portfolio

REPORTS = Path(__file__).resolve().parents[1] / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
OUT_SUFFIX = __import__("os").environ.get("OUT_SUFFIX", "")
RES = REPORTS / f"param_sweep_results{OUT_SUFFIX}.csv"
BASE = dict(hold=48, stop=2.0, take_profit=5.0, frenzy="无", score="无")


# ---------- 工具 ----------

def oos_monthly_returns(eq):
    e = eq.loc[(eq.index >= psb.IS_END) & (eq.index < psb.OOS_END)]
    if len(e) < 3:
        return None
    m = e.resample("ME").last()
    return m.pct_change().dropna()


def block_bootstrap(monthly, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    vals = monthly.to_numpy()
    n = len(vals)
    sh, rt = np.empty(B), np.empty(B)
    for b in range(B):
        v = vals[rng.integers(0, n, n)]
        sh[b] = v.mean() / v.std() * np.sqrt(12) if v.std() > 0 else np.nan
        rt[b] = np.prod(1.0 + v) - 1.0
    return sh, rt


def ci(x):
    return float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))


def gate_mask(ctx, f_name, s_name):
    df, f_thr, s_thr = ctx["df"], ctx["f_thr"], ctx["s_thr"]
    fmap = f_thr[f_name]
    f_series = df["d"].map(fmap) if fmap is not None else pd.Series(-np.inf, index=df.index)
    if s_name == "无":
        s_series = pd.Series(-np.inf, index=df.index)
    else:
        month = df["t"].dt.to_period("M").astype(str)
        s_series = month.map(s_thr[s_name]).fillna(-np.inf)
    return (df["frenzy"] >= f_series) & (df["score"] >= s_series)


# ---------- A. 秩相关 ----------

def part_a(df):
    print("== A. 秩相关与网格分布 ==")
    isoos = df[["sharpe_is", "sharpe_oos", "ret_is", "ret_oos"]].dropna()
    rho, p = stats.spearmanr(isoos["sharpe_is"], isoos["sharpe_oos"])
    print(f"  Spearman(sharpe_is, sharpe_oos) = {rho:+.3f}  (n={len(isoos)})")

    # 置换 p 值：打乱 OOS 夏普，重算 Spearman
    rng = np.random.default_rng(0)
    obs = rho
    null = np.empty(2000)
    for i in range(2000):
        y = rng.permutation(isoos["sharpe_oos"].to_numpy())
        null[i] = stats.spearmanr(isoos["sharpe_is"].to_numpy(), y).statistic
    p_perm = float((null >= obs).mean())
    print(f"  置换 p(单边, H1: ρ>0) = {p_perm:.3f}")

    rho_r, _ = stats.spearmanr(isoos["ret_is"], isoos["ret_oos"])
    print(f"  Spearman(ret_is, ret_oos)   = {rho_r:+.3f}")

    sh = df["sharpe_oos"]
    print(f"  OOS 夏普 >0 占比 {float((sh>0).mean()):.0%}  中位 {sh.median():+.2f}  "
          f"最大 {sh.max():+.2f} 最小 {sh.min():+.2f}")
    best_is = df.loc[df["sharpe_is"].idxmax()]
    print(f"  IS 最优夏普 = {best_is.sharpe_is:+.3f}，其 OOS 夏普 = {best_is.sharpe_oos:+.3f}，"
          f"OOS 排名 {(df.sharpe_oos >= best_is.sharpe_oos).sum()}/{len(df)}")
    return rho, p_perm


# ---------- B. 块自举 ----------

def part_b(ctx, df, B=2000):
    print("\n== B. 块自举 95% CI（OOS 窗口，按月分块） ==")
    best_is = df.loc[df["sharpe_is"].idxmax()]
    best_oos = df.loc[df["sharpe_oos"].idxmax()]
    cfgs = [
        ("基线 48/2/5", dict(hold=48, stop=2.0, tp=5.0, f="无", s="无")),
        ("IS 最优", dict(hold=int(best_is.hold), stop=float(best_is.stop),
                         tp=float(best_is.take_profit), f=best_is.frenzy, s=best_is.score)),
        ("OOS 最优", dict(hold=int(best_oos.hold), stop=float(best_oos.stop),
                          tp=float(best_oos.take_profit), f=best_oos.frenzy, s=best_oos.score)),
    ]
    for tag, c in cfgs:
        row = psb.run_combo(ctx, c["hold"], c["stop"], c["tp"], c["f"], c["s"],
                            return_equity=True)
        if row is None or "__equity" not in row:
            print(f"  {tag}: 无结果"); continue
        mr = oos_monthly_returns(row["__equity"])
        if mr is None or len(mr) < 4:
            print(f"  {tag}: OOS 月数不足"); continue
        sh, rt = block_bootstrap(mr, B)
        s_ci = ci(sh); r_ci = ci(rt)
        pt_sh = float(mr.mean() / mr.std() * np.sqrt(12)) if mr.std() > 0 else np.nan
        pt_rt = float(np.prod(1.0 + mr) - 1.0)
        print(f"  {tag}: OOS 夏普 {pt_sh:+.2f}  CI[{s_ci[0]:+.2f}, {s_ci[1]:+.2f}]  "
              f"| 收益 {pt_rt:+.1%}  CI[{r_ci[0]:+.1%}, {r_ci[1]:+.1%}]  (月数 {len(mr)})")


# ---------- C. 置换检验 ----------

def part_c(ctx, df, perms=500):
    print(f"\n== C. 置换检验：IS 最优门控 vs 等量随机抽样（{perms} 次） ==")
    best_is = df.loc[df["sharpe_is"].idxmax()]
    hold, stop, tp = int(best_is.hold), float(best_is.stop), float(best_is.take_profit)
    f_name, s_name = best_is.frenzy, best_is.score
    mask = gate_mask(ctx, f_name, s_name)
    k = int(mask.sum())
    print(f"  IS 最优 = {hold}/{stop}/{tp}/{f_name}/{s_name}，门控后事件 {k}")

    def one(sub):
        p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=stop,
                               take_profit_atr=tp, hold_bars=hold, signal_exit=True)
        out = portfolio.run(sub, ctx["bars"], p, entry_mode="confirm")
        if out is None:
            return None
        mr = oos_monthly_returns(out["equity"]["equity"])
        if mr is None or len(mr) < 4:
            return None
        return float(mr.mean() / mr.std() * np.sqrt(12)) if mr.std() > 0 else np.nan

    # 真实值（门控）
    real = one(ctx["df"][mask])
    print(f"  真实 OOS 夏普 = {real:+.2f}" if real is not None else "  真实值缺失")

    df_all = ctx["df"]
    rng = np.random.default_rng(42)
    null = []
    for i in range(perms):
        idx = rng.choice(len(df_all), size=k, replace=False)
        sub = df_all.iloc[np.sort(idx)]
        v = one(sub)
        if v is not None:
            null.append(v)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{perms}", flush=True)
    null = np.array(null)
    if real is not None:
        pct = float((null >= real).mean())
        print(f"  零分布: 中位 {np.median(null):+.2f}  P95 {np.percentile(null,95):+.2f}  "
              f"最大 {null.max():+.2f}")
        print(f"  真实值位于零分布第 {pct:.1%} 分位（H3: 需 ≥95% 才算门控有价值）")


# ---------- D. 核心七项 + 两项诊断（含块自举 CI） ----------
# 口径（D016）：核心 = 净收益、胜率、盈亏比、频率、每笔期望、收益/回撤、最大回撤+恢复天数；
# 夏普、Sortino 只算不用，作为诊断项。
# 核心项全部从窗口内净值/交易算；区间由月度收益有放回重抽 B 次得到（恢复天数用月路径近似）。

CORE = [("ret", "净收益"), ("win", "胜率"), ("payoff", "盈亏比"), ("freq", "频率"),
        ("avg_ret", "每笔期望"), ("calmar", "收益/回撤"), ("mdd", "最大回撤"),
        ("rec", "恢复天")]
DIAG = [("sharpe", "夏普"), ("sortino", "Sortino")]
PCT_KEYS = {"ret", "mdd", "avg_ret", "win"}


def window_monthly(eq, start, end):
    e = eq.loc[(eq.index >= start) & (eq.index < end)]
    if len(e) < 3:
        return None
    m = e.resample("ME").last()
    return m.pct_change().dropna()


def _path_stats(cum, n):
    ret = float(cum[-1] - 1.0)
    cagr = float((1.0 + ret) ** (12.0 / n) - 1.0) if ret > -1 else np.nan
    dd = cum / np.maximum.accumulate(cum) - 1.0
    mdd = float(dd.min())
    if mdd < 0:
        k = int(dd.argmin())
        prior = float(np.maximum.accumulate(cum)[k])
        after = np.where(cum[k + 1:] >= prior)[0]
        rec_m = float(after[0] + 1) if len(after) else np.nan
    else:
        rec_m = 0.0
    return ret, cagr, mdd, rec_m


def path_point(mr):
    v = mr.to_numpy(); n = len(v)
    cum = np.cumprod(1.0 + v)
    ret, cagr, mdd, rec_m = _path_stats(cum, n)
    sd = float(v.std(ddof=1))
    ddev = float(np.sqrt((np.minimum(v, 0.0) ** 2).mean()))
    return {
        "ret": ret, "cagr": cagr, "mdd": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else np.nan,
        "rec": rec_m * 30.44 if np.isfinite(rec_m) else np.nan,
        "sharpe": float(v.mean() / sd * np.sqrt(12)) if sd > 0 else np.nan,
        "sortino": float(v.mean() / ddev * np.sqrt(12)) if ddev > 0 else np.nan,
    }


def path_bootstrap(mr, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = mr.to_numpy(); n = len(v)
    keys = ["ret", "calmar", "mdd", "rec", "sharpe", "sortino"]
    out = {k: np.empty(B) for k in keys}
    for b in range(B):
        x = v[rng.integers(0, n, n)]
        cum = np.cumprod(1.0 + x)
        ret, cagr, mdd, rec_m = _path_stats(cum, n)
        sd = float(x.std(ddof=1))
        ddev = float(np.sqrt((np.minimum(x, 0.0) ** 2).mean()))
        out["ret"][b] = ret
        out["mdd"][b] = mdd
        out["calmar"][b] = cagr / abs(mdd) if mdd < 0 else np.nan
        out["rec"][b] = rec_m * 30.44 if np.isfinite(rec_m) else np.nan
        out["sharpe"][b] = x.mean() / sd * np.sqrt(12) if sd > 0 else np.nan
        out["sortino"][b] = x.mean() / ddev * np.sqrt(12) if ddev > 0 else np.nan
    return out


def six_bootstrap(mr, B=2000, seed=0):
    return path_bootstrap(mr, B, seed)


def _fmt(k, v):
    if v is None or not np.isfinite(v):
        return "n/a"
    if k in PCT_KEYS:
        return f"{v:+.1%}"
    if k == "rec":
        return f"{v:.0f}"
    if k == "freq":
        return f"{v:.0f}"
    return f"{v:+.2f}"


def part_d(ctx, df, B=2000):
    """核心七项 + 诊断两项。点估计用日频净值（与 param_sweep_results.csv 同源），
    区间用月度收益有放回重抽（近似）。"""
    import json

    print("\n== D. 核心七项 + 诊断两项（区间：月度自举 95%） ==")
    best_is = df.loc[df["sharpe_is"].idxmax()]
    best_oos = df.loc[df["sharpe_oos"].idxmax()]
    cfgs = [
        ("基线 48/2/5", dict(hold=48, stop=2.0, tp=5.0, f="无", s="无")),
        ("IS 最优", dict(hold=int(best_is.hold), stop=float(best_is.stop),
                         tp=float(best_is.take_profit), f=best_is.frenzy, s=best_is.score)),
        ("OOS 最优", dict(hold=int(best_oos.hold), stop=float(best_oos.stop),
                          tp=float(best_oos.take_profit), f=best_oos.frenzy, s=best_oos.score)),
    ]
    windows = [("IS", psb.WINDOW_START, psb.IS_END), ("OOS", psb.IS_END, psb.OOS_END)]
    records = []
    for tag, c in cfgs:
        row = psb.run_combo(ctx, c["hold"], c["stop"], c["tp"], c["f"], c["s"],
                            return_equity=True)
        if row is None or "__equity" not in row:
            print(f"  {tag}: 无结果"); continue
        eq = row["__equity"]
        mask = gate_mask(ctx, c["f"], c["s"])
        p = portfolio.BtParams(pos_pct=0.10, max_pos=10, stop_atr=c["stop"],
                               take_profit_atr=c["tp"], hold_bars=c["hold"], signal_exit=True)
        out = portfolio.run(ctx["df"][mask], ctx["bars"], p, entry_mode="confirm")
        if out is None:
            print(f"  {tag}: 回测无结果"); continue
        trades = out["trades"]
        for wname, a, b_ in windows:
            mr = window_monthly(eq, a, b_)
            sm = psb.slice_metrics(out["equity"], trades, p, a, b_)
            if mr is None or len(mr) < 4 or sm is None:
                print(f"  {tag} {wname}: 样本不足"); continue
            pt = {k: sm.get(k) for k in ["ret", "win", "payoff", "freq", "avg_ret",
                                         "calmar", "mdd", "sharpe", "sortino"]}
            pt["rec"] = sm["rec_days"]
            boot = path_bootstrap(mr, B)
            tr = trades[(trades.entry_time >= a) & (trades.entry_time < b_)] if trades is not None else None
            if tr is not None and len(tr):
                vals = tr["ret"].to_numpy()
                rng = np.random.default_rng(7)
                draws = vals[rng.integers(0, len(vals), (B, len(vals)))]
                boot["win"] = (draws > 0).mean(axis=1)
                boot["avg_ret"] = draws.mean(axis=1)
                aws = np.array([d[d > 0].mean() if (d > 0).any() else np.nan for d in draws])
                als = np.array([d[d < 0].mean() if (d < 0).any() else np.nan for d in draws])
                boot["payoff"] = np.abs(aws / als)
                boot["freq"] = np.full(B, np.nan)
            else:
                for k in ("win", "avg_ret", "payoff"):
                    boot[k] = np.full(B, np.nan)
            rec = {"cfg": tag, "window": wname, "months": len(mr),
                   "n_trades": int(sm["n"])}
            parts = []
            for k, name in CORE:
                lo, hi = ci(boot.get(k, np.full(B, np.nan)))
                rec[k] = pt.get(k)
                rec[k + "_ci"] = ([lo, hi] if (k != "rec" and np.isfinite(lo))
                                  else [None, None])
                v = pt.get(k)
                if k == "freq":
                    parts.append(f"{name} {_fmt(k, v)}/年")
                elif k == "rec":
                    parts.append(f"{name} {'未恢复' if not np.isfinite(v) else f'{v:.0f}天'}")
                else:
                    parts.append(f"{name} {_fmt(k, v)} [{_fmt(k, lo)}, {_fmt(k, hi)}]")
            for k, name in DIAG:
                lo, hi = ci(boot[k])
                rec[k] = pt.get(k); rec[k + "_ci"] = [lo, hi]
            records.append(rec)
            print(f"  {tag} {wname} ({len(mr)} 月/{rec['n_trades']} 笔): " + " | ".join(parts))

    out_path = REPORTS / f"six_metrics{OUT_SUFFIX}.json"
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2))
    print(f"  写出 {out_path}")
    return records


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perms", type=int, default=0)
    ap.add_argument("--six", action="store_true", help="只跑 D：六指标带 CI")
    args = ap.parse_args()

    df = pd.read_csv(RES)

    ctx = psb.build_context()
    if args.six:
        part_d(ctx, df, args.boot)
        return
    part_a(df)
    if not args.skip_b:
        part_b(ctx, df, args.boot)
    if args.perms > 0:
        part_c(ctx, df, args.perms)


if __name__ == "__main__":
    main()
