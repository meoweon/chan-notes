"""从事件表 + 锚点追踪表生成 BSP 分类/占比表。

输入：
- data/events/crypto_21_v1.parquet（事件行；每行一个 (code, bi, type2str) 首次出现）
- data/events/crypto_21_v1_anchor_py.parquet（可选；track_bsp_anchor.py 产物）

输出：
- reports/bsp_taxonomy_stats.md

用法：
    .venv/bin/python scripts/bsp_taxonomy_stats.py
"""
import argparse
from pathlib import Path

import pandas as pd

TYPE_ORDER = ["1", "1p", "2", "2s", "3a", "3b", "2,3b", "2s,3b", "1,2s"]


def _pct(x):
    return f"{x:.2%}" if pd.notna(x) else "-"


def _object_table(first: pd.DataFrame, anchor: pd.DataFrame | None) -> str:
    rows = []
    for t in TYPE_ORDER:
        g = first[first.bsp_type == t]
        if g.empty:
            continue
        dead = g.clear_bar >= 0
        life = (g.clear_bar - g.bar_idx)[dead]
        late = (g.delay_bars > 0) & (g.bar_idx > 200)
        moved = "-"
        if anchor is not None:
            a = g.merge(anchor, on=["code", "bsp_bi_idx"], how="left")
            alive = a.clear_bar == -1
            moved = _pct((a.n_changes.fillna(0) > 0)[alive].mean())
        rows.append(
            f"| {t} | {len(g)} | {_pct(len(g) / len(first))} | "
            f"{g.is_buy.mean():.1%}/{1 - g.is_buy.mean():.1%} | "
            f"{g.survive_144.mean():.1%} | {life.median():.0f} | "
            f"{late.sum()} ({late.mean():.2%}) | {g.conf_ok.mean():.1%} | {moved} |"
        )
    head = (
        "| 类型 | 对象数 | 占比 | 买/卖 | 存活率 | 死亡中位寿命 | 晚首次 | 确认率 | 存活者锚点挪过 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    return head + "\n" + "\n".join(rows)


def _event_table(df: pd.DataFrame) -> str:
    rows = []
    for t in TYPE_ORDER:
        g = df[df.bsp_type == t]
        if g.empty:
            continue
        late = (g.delay_bars > 0) & (g.bar_idx > 200)
        rows.append(
            f"| {t} | {len(g)} | {_pct(len(g) / len(df))} | "
            f"{g.is_buy.mean():.1%}/{1 - g.is_buy.mean():.1%} | "
            f"{g.survive_144.mean():.1%} | {late.sum()} ({late.mean():.2%}) | "
            f"{g.conf_ok.mean():.1%} |"
        )
    head = (
        "| 类型 | 事件行 | 占比 | 买/卖 | 存活率 | 晚首次 | 确认率 |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    return head + "\n" + "\n".join(rows)


def _lifecycle(df: pd.DataFrame, first: pd.DataFrame, anchor: pd.DataFrame | None) -> str:
    lines = []
    if anchor is not None:
        a = df.merge(anchor, on=["code", "bsp_bi_idx"], how="left")
        a["moved"] = a.n_changes.fillna(0) > 0
        alive = a.clear_bar == -1
        n_a = int((alive & ~a.moved).sum())
        n_d = int((alive & a.moved).sum())
        n_b = int((~alive).sum())
        lines.append(f"| A 存活且锚点未动 | {n_a} | {_pct(n_a / len(a))} |")
        lines.append(f"| D 存活但锚点动过 | {n_d} | {_pct(n_d / len(a))} |")
        lines.append(f"| B 首次消失 | {n_b} | {_pct(n_b / len(a))} |")
        lines.append(f"| A+D+B | {n_a + n_d + n_b} | 100.00% |")
    late = (first.delay_bars > 0) & (first.bar_idx > 200)
    n_late = int(late.sum())
    c_denom = int((df.bar_idx > 200).sum())
    lines.append(f"| C 晚首次（正交） | {n_late} | {_pct(n_late / c_denom)} |")
    extra = (df.groupby(["code", "bsp_bi_idx"]).cumcount() > 0) & (df.delay_bars > 0)
    lines.append(f"| E 类型追加（隐藏） | {int(extra.sum())} | {_pct(extra.mean())} |")
    if anchor is not None:
        a["gap"] = (a.last_bar - a.first_bar + 1) - a.n_seen
        n_f = int((a.gap > 0).sum())
        lines.append(f"| F 消失又回来（隐藏） | {n_f} | {_pct(a.gap.gt(0).mean())} |")
    else:
        lines.append("| F 消失又回来（隐藏） | - | 需锚点追踪表 |")
    head = "| 类别 | 数量 | 占事件行 |\n|---|---:|---:|"
    return head + "\n" + "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/events/crypto_21_v1.parquet")
    ap.add_argument("--anchor", default="data/events/crypto_21_v1_anchor_py.parquet")
    ap.add_argument("--out", default="reports/bsp_taxonomy_stats.md")
    args = ap.parse_args()

    df = pd.read_parquet(args.events)
    df = df.sort_values(["code", "bsp_bi_idx", "bar_idx", "event_time"]).reset_index(drop=True)
    first = df.groupby(["code", "bsp_bi_idx"], as_index=False).head(1)

    anchor = None
    if Path(args.anchor).exists():
        anchor = pd.read_parquet(args.anchor)
        anchor = anchor.rename(columns={"bi_idx": "bsp_bi_idx"})

    md = [
        "# BSP 分类与占比（脚本输出）",
        "",
        f"- 事件行：{len(df)}；唯一 BSP 对象：{len(first)}",
        f"- 方向：买 {df.is_buy.mean():.2%} / 卖 {1 - df.is_buy.mean():.2%}（事件行）",
        f"- 存活率：{df.survive_144.mean():.2%}；确认率：{df.conf_ok.mean():.2%}",
        "",
        "## 唯一 BSP 对象",
        "",
        _object_table(first, anchor),
        "",
        "## 事件行（模型样本）",
        "",
        _event_table(df),
        "",
        "## 生命周期",
        "",
        _lifecycle(df, first, anchor),
        "",
        "说明：C 是剔除 warm-up（bar_idx<=200）后的晚首次；分母在“唯一对象 / 事件行”两种口径下都给出。"
        "F 由锚点追踪表的 (last_bar-first_bar+1-n_seen)>0 判定（对象在首末出现之间有过缺席），按事件行统计。"
        "A/D/B 的锚点变化来自 track_bsp_anchor.py（Python vendor 引擎）；若未跑，则只输出 C/E/F。",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
