# crypto-breadth（已作废）

> **作废日期**：2026-11-09
> **作废原因**：用户判定「不好用」，同日从 `repos.toml` 移除并删除 vendor 快照；相关决策 D019 亦已标注作废。
> **替代文档**：无（该仓不再维护）；当前仓库清单见 仓库全景。
> 以下原文按「作废不删」保留作留痕，结论不再采信。

- 类别：05_回测与实盘
- 角色：策略研究参考（横截面动量 / 市场宽度闸门）
- 许可：无（GitHub API 返回 license: null，快照内无 LICENSE）
- 上游：`phuazz/crypto-breadth`（pin `52f22bff`，2026-11-09 daily refresh）
- 快照：`vendor/05_回测与实盘/crypto-breadth`
- 代码规模：45 个 py/js 文件约 11,630 行；快照落盘 9.3 MB，入库 94 个文件

## 一句话

非缠论路线的加密横截面动量策略：25 币滚动流动性池 + 50 日均线宽度闸门分档仓位 + 30/90/180 日动量选前 4，周一调仓、每日退出覆盖；带预注册评审、试验台账与复现测试，是本仓「可证伪流程」的对照样本，不是可采信的绩效证据。

## 已核实（E1）

- 研究底稿是 `data/prices.parquet`，由 `scripts/fetch_data.py` 本地 bootstrap，CI 无法刷新（币安对美国 runner 返回 451）——`DATA_INTEGRITY_POLICY.md:17`；point-in-time 池与幸存者风险的约束见 `:21-30`。
- 策略口径可逐行核对：宽度分母是当日「可投资且已有完整 MA」的币种数，不是全池（`scripts/backtest.py:172-183`）；按 0.30/0.50/0.70 阈值映射 0/30/60/100% 仓位（`:186-193`、`:88-89`）；动量是 30/90/180 日风险调整收益的等权均值（`:196-212`）；入场要求价 > 自身 MA 且 MA 上行、出场只要价 < MA，均为 T 日收盘出信号、T+1 收盘成交（`:215-239`）；只在周一生成目标权重，非调仓日置空由回测循环前向填充、随价格漂移（`:261-293`）；费用 10bps/边按成交额扣（`:69`、`:367`、`:381`）。
- 防前视有测试兜底：`tests/test_no_lookahead.py:23-87` 四个断言分别 pin 了调仓与日退出的 T+1 滞后，以及动量/宽度对未来价格扰动的不变性。静态检查未运行任何上游代码。
- v3.1 与 v3.2 只差单名上限：`build_target_weights` 的 cap 默认 None（`:265`），生产 `Params` 为 0.34（`:85`）；`tests/test_v31_reproducibility.py:34-126` 双向静态保证——研究 harness 不传 cap 以复现存档记录，生产调用点必须传 cap 以防静默回落到 v3.1。
- 评审产物与 README 口径一致：`results/phase_b_review.json` 的 full Sharpe 1.3514 / MaxDD −0.4478 / OOS 1.4285 对应 README 的 v3.1「1.35 / −44.8%」；DSR 在 N=10~200 阶梯上均 ≥0.9988。
- 闸门消融产物实测：`results/phase_c4_gate.json` 中 breadth vs none 为 Sharpe 1.3514 vs 0.7424（+0.609）、MaxDD −0.4478 vs −0.8294（+38.2pp），vs BTC-200d 闸门 +0.173；分档升级本身几乎无增益（binary gate 1.3186）。
- 试验台账 `results/trial_registry.jsonl` 共 104 条、配置 id 全不重复，arm 分布 B 64 / C3b 20 / C2 5 / C3a 5 / C4 5 / E 5；DSR 用的 N_BASE=50 是 `scripts/research/phase_b_review.py:313` 的硬编码，`results/phase_b_review.md:81-87` 说明它来自敏感性/网格/结构阶梯的估算，不是从台账自动统计。104 落在其 N=10~200 稳健区间内，不改变 DSR 结论，但口径要写清。
- 数据事故留痕完整：`results/survivorship_audit.md:17-29` 记 25 币无一提前退市、40 个 2019–2021 有真实流动性的币全部缺席、候选池不足真实池 40%；`:69-109` 记 2026-07-18 发现 LUNA 列是三身份拼接（含 +177,399% 伪收益），先跑反事实（tier 只变 5 天、净值 bit-identical）再 purge 1509 行并加冻结测试。
- 快照入库 94 个文件：`data/prices.parquet` 与 `data/backtest_equity.parquet` 被本仓 `.gitignore:92` 的 `*.parquet` 忽略，落盘但未入库（sync 提交 345e82c）。
- 上游按日 CI 刷新 data：GitHub commits API 最近 12 条提交全为 `ci: daily refresh`；`.github/workflows/tests.yml:3-5` 说明数据刷新不触发测试。

## 作者声称（E2）

- README 自述 "Research-grade — not deployed, no live track record"（`README.md:5`）；v3.2 全样本 Sharpe 1.359 / MaxDD −39.5%（`:89`），样本外 Sharpe 1.456（`:119`）。本仓未复算，不采信为结论。
- `results/phase_b_review.md:8-18` 的 KEEP 结论建立在事后移除 −30% MaxDD 上限之上，按原预注册规则是 MODIFY / do-not-deploy；原文自述，未独立核实。
- README 自述 25 币池是 hindsight-selected，headline 数字带乐观的幸存者选择偏差，应读作上界（`README.md:32-38`）。

## 风险 / 坑

- **`results/` 与 dashboard 全部是 v3.1 口径，实际 live 是 v3.2**（加 34% 单名上限）；引用任何数字前先读 `README.md:72-89` 的版本说明。
- 25 币池不含 2019–2021 已退市/衰落的币种，宽度闸门只能消除池内偏差；headline 是上界（`README.md:32-38`、`DATA_INTEGRITY_POLICY.md:21-30`）。
- 消融数字不可混用：`README.md:43` 的「+0.25 Sharpe / +30pp MaxDD」对应 `scripts/research/diagnostics.py:65` 的「MA warmup 后才满仓」口径，与 `results/phase_c4_gate.json` 的 none arm（+0.609 / +38.2pp）定义不同；引用时必须注明是哪一次消融。
- DSR 的试验数是硬编码估算而非台账统计；虽然对 N=10~200 稳健，这个做法本身不符本仓「统计口径可复算」的要求。
- 预注册判据被事后修改（见上），与本仓「先写判据再跑」的纪律冲突，不能把它的 verdict 当 E1 结论。
- 无 LICENSE：只读、不运行、不复制代码进本仓、不对外分发，与 D010 对 LuckyCat / PoolPilot 的处置一致。
- 快照不含研究底稿 parquet；要复算需按 `DATA_INTEGRITY_POLICY.md:17` 自行取数。
- 上游每日 CI 刷新，pin 落后是常态；只在评审/结论变化时手动前进 pin，不追 daily commit。

## 与本仓关系

- 非缠论路线，标的、标签、评估口径与 chan-quant / mdk 都不同，信号与绩效不可直接比较。
- 值得借鉴的是研究纪律：预注册（RESEARCH_MEMO PR-1）、试验台账、分阶段评审、复现测试、作废留痕，与本仓 E1/E2 分级和预注册制度同构；LUNA 事故的「发现 → 反事实量化 → 手术 → 冻结测试」尤其可直接照搬。`phase_b_review.md` 的判据事后修改、DSR 试验数硬编码则是反例。
- 归入 05 类作策略/回测参考，与 LuckyCat / PoolPilot 一样只读不运行；不接交易所、不进回测主线。

## 待办

- [x] 核对试验计数与 DSR 口径（2026-11-09）：台账 104 条、DSR 用 N=50 硬编码估算并附 N=10~200 阶梯，结论对该区间不敏感；口径差异已记入「风险 / 坑」。
- [ ] 若要复算：先按数据策略取 `data/prices.parquet`，再跑 `tests/test_v31_reproducibility.py` 与 Phase B 复现脚本。
- [ ] 核对 v3.1 → v3.2 的 34% 上限对 MaxDD 结论的影响。
- [ ] 核对 `diagnostics.py` 与 `phase_c4_gate.json` 两种「无闸门」定义，统一 README 消融数字的引用口径。

## 变更记录

- 2026-11-09：登记 `repos.toml`、sync 快照（pin 52f22bff）、建卡（D019）。
- 2026-11-09：静态核查代码/测试/评审产物，补 E1 证据、消融口径与 DSR 口径差异，结掉试验计数待办；未运行上游代码。
