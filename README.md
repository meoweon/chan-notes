# chan-notes

**缠论结构 × Brooks 价格行为 × 机器学习的 A 股 30 分钟研究流水线。**

这个仓库记录的是**证据链**：哪些假设被证伪、哪些被修正、哪些在样本外还站得住。
所有结论都要求可复算、可证伪。不接实盘、不下单、不保存任何券商凭据。

> 数据：HS300 成分股，30 分钟 2020-01 ~ 2026-10 + 日线 2010-01 ~ 2026-10，前复权（baostock）
> 主样本：289 股 / 47,307 个 BSP 事件 / 8,646 个确认买点；特征 188 列（v4 基线 154 + dy_ 动力学 34）
> 结论截至：2026-11

## 结论摘要

**一句话：入场时刻，缠论结构能预测的只有结构本身（BSP 存活）；收益由外部资金流决定，入场时没有签名——但 A 股的"什么时候开机"可以外推，换到加密市场则不成立。**


| 问题                                   | 结论                                                                                                                            | 证据                                     |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 裸 BSP 买点有 alpha 吗                    | **没有**：BUY -6.0bp/笔（t=-1.54）                                                                                                  | `reports/experiment_log.md` #1         |
| 加工后（确认入场 + 144bar/3ATR + 无卖点离场）有正期望吗 | **账面上有**：2024–26 样本外累计 +50.5%、年化 15.6%、夏普 0.80、回撤 -21.2%、单笔 +58.3bp                                                           | `reports/strategy_v2_report.md`        |
| 这个期望可复制吗                             | **不能——全在尾部**：去掉 top5% 赢家后单笔期望转负；中位交易约 -2.4%                                                                                   | `reports/tail_test.csv`                |
| ML 能预测"这笔赚不赚"吗                       | **不能**：方向、大波、卖点、对齐收益等 5 次独立阴性，AUC 约 0.49–0.58                                                                                 | `reports/experiment_log.md`            |
| ML 能预测什么                             | **BSP 存活**：AUC 0.658（买）/ 0.689（卖），67 个月 100% ≥0.55                                                                            | `reports/experiment_log.md` #17        |
| regime 门控能外推吗                        | **A 股能**：2023 年底冻结阈值，2024–26 样本外夏普 1.21 vs 基线 0.80，回撤 -12.3% vs -21.2%。**69 币加密复测判负**（ΔSharpe -0.30 ~ -1.44，门开集样本外单笔 -15.0bp） | `reports/frenzy_frozen_oos.csv`；台账 #75 |


读法：前四行是"证伪"，后两行是"留存"。门控是本研究最强的 A 股结果（#19），但它在 69 币加密考卷上门线判死（#75），说明它捕捉的是 A 股特有的狂潮 regime，不是通用 alpha；且本质仍是尾部游戏——去 top5% 后期望转负，开机率约 40%，不是全天候策略。

## 流水线

```
baostock 30m/d ──► parquet 缓存 ──► framework 逐 bar 回放 ──► BSP 首次出现事件
                                                               │
                        ┌──────────────────────────────────────┤
                        ▼                                      ▼
              188 列因果特征（≤ t）                    前向标签（> t，物理隔离）
              · pa_  价格行为 54                     · ret / dir_ret / mfe / mae / stop
              · pt_  形态 15                         · pol_*  止损 + 时间退出
              · ch_  缠论 54                         · conf_* 确认入场
              · dy_  动力学 34                       · survive_144  BSP 存活
              · mkt_ 市场 8
              · sw_/rg_/bc_/bsw_/ff_/rf_/wg_  Brooks 23
                        │                                      │
                        └──────────────┬───────────────────────┘
                                       ▼
                          月度 walk-forward（XGBoost）
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
              AUC / 分位 lift     逐 bar 组合回测      特征体检
                                （单笔 10%、最多 10 仓）  （消融/冗余/重要性）
```

三条硬约束贯穿全仓：

1. **因果**：特征只读 `≤ t` 的数据；`t` 之后的数据只用于标签。
2. **单一实现**：标签、评估、回测共用 `chan_notes/core/policy.py` 的退出模拟，避免同一规则复制多份后走样。
3. **预注册**：新实验先写判据、数据、切分、成本与停止条件，再跑；判据不得由结果反选。



## 目录

```
chan_notes/               核心包
  config.py               路径、数据目录与引擎参数
  data/                   parquet 缓存与 baostock 抓取
  core/replay.py          逐 bar 回放、BSP 首次出现与存活追踪
  core/policy.py          唯一一份退出规则模拟（止损 / 时间退出 / 删失）
  features/               188 列因果特征（含 Brooks 移植实现）
  labels/                 前向标签与确认入场
  eval/                   AUC、bootstrap CI、分位 lift、期望值
  models/train.py         特征列管理与 XGBoost 训练
  gates/                  regime 门控（广度 / 狂热指数）
  backtest/portfolio.py   逐 bar 盯市组合回测
framework/                内置缠论算法（53 个文件，第三方组件见末节）
scripts/                  数据、实验、报告脚本（94 个，入口见下）
reports/                  研究报告、特征体检 CSV、实验台账（34 件）
vendor/LocalStore.py      本地 parquet 数据源适配器
tests/                    regime 门控口径的单元测试
```



## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/setup_framework.py    # 校验内置 framework 并安装数据源适配器
python scripts/fetch.py              # 抓 HS300 30m（断点续跑，首次约 2 小时）
python scripts/build_market.py       # 等权市场序列
python scripts/build_dataset.py --name events_main
python scripts/baseline.py --name events_main
python scripts/analyze_features.py events_main
python scripts/train_wf.py events_main wf_main c_pol_dir_ret conf_buy 1200 0.02
python scripts/run_backtest.py --name events_main --mode confirm --stop 1.5 --hold 48
```

数据与训练产物不进仓库：`data/`、`logs/`、`refs/` 均在 `.gitignore` 中，按上面的脚本重建。

## 主要脚本


| 环节    | 脚本                                                                                                                              |
| ----- | ------------------------------------------------------------------------------------------------------------------------------- |
| 数据    | `fetch.py`、`download_ashare.py`、`download_binance_futures.py`、`backfill_*.py`、`build_market.py`、`resample_daily.py`             |
| 数据集   | `build_dataset.py`、`build_swing_events.py`、`track_bsp_anchor.py`、`make_v5/v6/v7_*.py`                                           |
| 评估与训练 | `baseline.py`、`analyze_features.py`、`train_wf.py`、`train_wf_oos.py`、`train_wf_xgb.py`、`run_backtest.py`                         |
| 机制检验  | `tail_test.py`、`tail_analysis.py`、`tail_regime.py`、`frenzy_*.py`、`regime_*.py`、`placebo.py`、`sell_filter.py`、`adaptive_exit.py` |
| 报告    | `make_*_report.py`、`make_equity_*.py`、`case_study.py`、`btc5m_viewer.py`                                                         |
| 口径校验  | `test_prefix_invariance.py`、`audit_bsppoint_freeze.py`、`validate_bsppoint_profit.py`                                            |




## 数据源规则

- baostock 官方明确**禁止多连接**（多线程/多进程）访问。本仓抓取脚本一律单进程 + 请求间隔，不要并行化。
- parquet 布局：`data/bars/{freq}/{code}.parquet`，事件表在 `data/events/`。可用环境变量 `CHAN_NOTES_BARS` 覆盖数据根目录。
- 30 分钟数据起点 2020-01（baostock 限制），日线可回溯到 2010。



## 实验与报告

- `reports/experiment_log.md`：**75 条编号台账**，是本仓的总索引。每条记录数据版本、假设、方法、结果与失效条件。
- `reports/strategy_v1_report.md` / `strategy_v2_report.md`：入场与出场规则的完整结论。
- `reports/feature_label_catalog.md` / `feature_effectiveness.md` / `feature_selection.csv`：特征目录、有效性与冗余分析。目录正文停在 131 列的 v1 口径，Brooks 23 列与 dy_ 34 列的增量见台账 #15 / #16。
- `reports/bsppoint_audit_20261109.md`：对早期买卖点报告的口径审计（含自我纠错）。
- `reports/phase_a_tail_report.md`、`tail_*.csv`：尾部依赖检验。
- `reports/frenzy_*.csv`、`regime_*.csv`：A 股 regime 门控的参数、样本外与稳健性结果；69 币加密复测的完整判定在台账 #75。
- `reports/perfect_state_report_作废.html`：历史上的“筛选链路与上限对比”页（算法上限 / 机制上限 / 最优模型上限 P=R=90%）。已删去实盘曲线与模型读数，只保留三层理论层；5×ATR 止盈与 score≥0.5 筛选口径均已判死，勿当策略收益。

研究结论会保留被推翻的版本：作废的结论标记原因与日期，不删除原文。

## 加密生存模型（69 币）的成绩

> 口径：69 个 USDT 永续，1h，2020-01 ~ 2026-11-08；43,978 个 BSP 事件（剔 44 行删失），标签 `survive_144`，阳性率 46.93%；XGBoost 202 列、71 个月 walk-forward、阈值 0.5。下表是基线臂的 pooled 样本外读数。

| 指标 | 数值 | 大白话 |
|---|---:|---|
| AUC | 0.6992 | 随机抽一对（一个真存活、一个真失效的 BSP），模型给真存活打高分的概率约 70%；瞎猜 50%，满分 100% |
| 准确率 | 64.55% | 全部判断里对了 64.6%；阳性率 46.9%，全猜“不存活”是 53.1% |
| 召回率 | 56.61% | 100 个真存活事件，抓住 56.6 个，漏 43.4 个 |
| 精确率 | 63.54% | 模型说“会存活”100 次，63.5 次对、36.5 次误报 |

这条线的正式结论是**加特征没有增量**：41 列 Alpha158 真新子集消融判阴性（ΔAUC −0.00037，月聚类 95% CI [−0.00234, +0.00125]，41 列无一进重要性前 50），因此不再往特征侧加料。存活的判别力已经稳定，后续提升更可能在标签质量与事件识别本身。

### 一条交易观（经验判断，不是本仓数据结论）

买卖点的确定性与期望收益成反比：确定性越高，单笔利润空间越小；高期望来自少数低确定性事件，也就是“富贵险中求”。长期能在市场里活下来的做法，通常是拿确定性换较低的期望；黑天鹅来临时基本都会亏，靠风控认亏。少数赌徒可以靠黑天鹅暴富，但那不可复制。

