# E019 新信息维度：主动买卖拆分与资金费率能不能在 E017 之上再加一层

- 日期：2026-11-10 立项 / 2026-11-11 跑完
- 状态：**已跑完并判定 = `marginal_new_info`**（`m_both − m0` Δacc **+0.126pp**，未过 +1.0pp 门；C2 归因 = 增量全在 taker 组、资金费率无额外贡献）；S1/S2/S3/S4/S5/S6 六道硬门全部落盘，另记 4 条偏差（§2 末）
- 关联仓库：本仓（bars 只读）；新数据落 `data/futures_flow/`（新目录，不覆盖任何在案产物）
- 关联决策 / 前置实验：判据对象 = [E017](E017-成交量回归模型.md) 冻结的 m1（0.67270）；有效性 / 判定行集 / 评分函数 = [E014](E014-成交量方向基线.md) 逐字继承；D042 第 5 条（尺子同源）、D043 第 1 条（两档参照）、D044 第 1/2/4 条（点验落盘 / 同一表达式 / 协议重冻窗口）、D045 第 4 条（晚上市与无历史处置）
- 代码：`exp/E019-new-info-dimension/`（探针已建：`code/probe_newinfo.py` → `out/probe/`）

## 1. 预注册（运行前填）

### 问题

E017 实测：在只喂「量的形状 + 收益率 + 日历 + 币种身份」的同一张 OOS 考卷上，收益率组总 gain 仅 0.074%、币种身份 0.11%、小时 0.22% —— **同源加列已无量可挖**。本实验问一个可回答的问题：**换成真正不同源的两类信息——(a) 同一小时成交量的主动买/主动卖拆分（taker flow），(b) 永续合约已结算资金费率——能否把 OOS 方向准确率在 E017 m1 的 0.67270 之上再推高 ≥1.0pp？**

### 假设

- H1（跑前预期，E3，可证伪）：taker 拆分单独 +0.5~+2.0pp、资金费率单独 +0.2~+1.0pp、两者同喂 +0.7~+2.5pp；**70% 仍大概率达不到**（0.6727 + 2.5pp = 0.698）。理由（E1 探针）：主动买占比在正常币上有实分布（p01~p99 = 0.27~0.67），而 E014 的 a5 与 E017 m1 都**只看总量**，结构上看不见这一维。
- H0（零假设，必须能接受）：新维度 Δ ≤ 0 —— 则「加新信息」这条路和「加同类特征」一样关闭，照实写阴性，不做补救（D042 第 1 条）。

### 判据

- **C1 主判定（跑前写死）**：`m_both − m0` 的 OOS 方向 accuracy（逐币等权，69 币）**Δacc > +0.010** 且 pooled 同向 且 balanced accuracy 更高 ⇒ 判 `new_info_adds_direction`；Δacc ≤ 0 ⇒ 判 `new_info_dead_end`；0 < Δacc ≤ 0.010 ⇒ 判 `marginal`。门槛取 1.0pp（不是 E017 的 2.0pp）的理由写明：对手从「零训练基线」换成「已在 67.27% 的模型」，而 E017 实测币簇 bootstrap 的 Δ 半宽约 ±0.28pp，1.0pp ≈ 3.5 个半宽，已足够排除抽样噪声；同一门槛同时用于披露口径（D044 第 2 条：分子 = 逐币等权 acc 差，分母 = 69 币，判定行集固定 1,686,328 行）。
- **C2 分维度归因（不进 C1，但必须逐条报）**：`m_taker`、`m_funding` 各自与 `m0` 的 Δacc / Δbacc / Δpooled，加新列的 gain 份额、以及**币簇层面**的正负币数与中位 Δ——用来回答「是哪一类信息在出力，还是只有量在出力」。若 `m_taker` 与 `m_both` 的 Δ 差 < 0.3pp，表述为「增量归属 taker，资金费率无额外贡献」，不得写成「两者都有贡献」。
- **C3 硬自检（先于任何判定数字）**：
  - **S1 拼接逐位对拍**：对全部 69 币，新抓的 1h klines 归档与 `data/bars/1h/*.parquet` 在 `volume / close / amount` 三列**差值必须为 0.0**（`amount` 对 `quote_volume`），且时间戳双向无孤儿行（`n_only_bars == n_only_zip == 0`）。探针已在 12 个币-月上实测 `max_abs_dev = 0.0`，本条要求全量成立，否则**硬失败停机**——因为整条推理依赖「taker 列与 bars 同源同戳」。
  - **S2 尺子同源**：`m0`（不加新列的臂）必须与 E017 冻结产物**逐位相同**（`out/metrics_by_coin.csv` 全列 max_abs_dev == 0.0），并顺带重算 E014 的 a5 == 0.62382（≤1e-9）。差值不为 0 即说明特征帧或行集漂了，判据作废重跑。
  - **S3 防前视**：① 资金费率只允许 `funding_time <= t` 的**已结算**行（决策时刻是 bar t 收盘，故 t+1 整点那一笔**本可用但一律不用**，宁保守）；② 特征构建器过 E017 同款两测试（截断后重算逐位不变 + 未来行扰动不变），币覆盖 BTC/ETH/JTO/SOL × volume/amount = 8 组；③ 资金费率区间**从归档 `funding_interval_hours` 列读取**，禁止硬编码 8h——探针实测 TON 2024-03 同月内 4h 与 8h 并存、2025-10 全为 4h。
  - **S4 覆盖与新鲜度（分母写死）**：落盘「过去 24h 已结算费率之和为 0 的判定行份额」「距上次结算小时数分布」「无资金费率归档的币-月清单」。探针口径的分母 = 该币当月有效 bar（`volume>0`）行数；全量口径的分母 = 1,686,328 判定行，两者不得混用（D044 第 2 条）。
  - **S5 数据冻结**：跑前把 69 个 bars 文件的 sha256（沿用 E014 清单）与 E014 / E017 两个协议 sha256 写进 `protocol.json`，跑后断言全部不变；新数据只写 `data/futures_flow/`，**不改 `data/bars/`**（并发会话 E016 正在读 bars）。
  - **S6 确定性**：全量重跑一遍，`metrics_by_coin.csv` / `metrics_summary.json` / `feature_importance.csv` / `funding_join.csv` / `protocol.json` 的 sha1 两遍逐位一致。
- **停止条件**：抓取 + 训练 + 判定一次跑完即停；不扫参、不换列集、不加第二个标签口径、不补救；抓取墙钟上限 60 分钟、训练判定上限 30 分钟，超限记录并停。
- **本期明确不做**：逐笔 aggTrades 级 CVD（实测 BTC 单币单月 **522.7 MB**，69 币全历史 TB 级，磁盘仅剩 78 GB）；盘口 bookTicker（实测单币单月 **6.95 GB**）；持仓量 open interest（`fapi/openInterestHist` 本机返回 HTML 错误页，且官方只留 ~30 天历史）；蒸馏（用户 2026-11-10 明确「目的不是蒸馏」）。

### 数据

- 来源：`https://data.binance.vision/data/futures/um/monthly/`（公开归档、无凭据、TLS 校验开启；本机实测 `api.binance.com` / `fapi.binance.com` / vision 三端点均 200）。两类文件：
  - `klines/<SYM>/1h/<SYM>-1h-<YYYY-MM>.zip`：12 列归档，取现脚本**丢掉的第 8/9/10 列** `count / taker_buy_volume / taker_buy_quote_volume`（丢弃位置：`scripts/download_binance_futures.py:73-75` 用 `header=None` 读、`:111-121` 只保留 `raw[0..7]`）。实测单文件 zip 均值 **30.4 KB**（13 个文件 395 KB）。
  - `fundingRate/<SYM>/<SYM>-fundingRate-<YYYY-MM>.zip`：列为 `calc_time, funding_interval_hours, last_funding_rate`（**带表头**，与 klines 归档不同，必须嗅探）。实测单文件均值 **822 B**。
- 覆盖实测（探针，`out/probe/`）：BTC 资金费率自 **2020-01-01** 起、每月 93 笔（8h）；69 币中 2024-01 有 4 币 404（JUP/RENDER/STRK/TON，均为晚上市，改测其上市月即 200）；**TON 2026-10 归档 404 且当月 bars 全 720 根 `volume==0`**（退市尾段）。
- bar 有效性 / 判定行集：与 E014、E017 **完全同一套**——评分对 Δt 恰好 1h、双端 `volume > 0`、平局剔除，OOS 判定行 **1,686,328** 行；由 `import baselines` 逐字继承，不重写。新数据不引入第二套有效性。
- 数据可得时点：taker 列随 bar 收盘即得（同归档同戳）；资金费率只取 `calc_time <= t` 的已结算值（S3①）。
- 一致性旁证（探针）：15m 归档聚合到小时后，`volume` 与 `taker_buy_volume` 对 1h 归档的相对偏差最大值 **2.2e-16**（浮点舍入量级，744/744 小时无一行 >1e-6）⇒ taker 列在归档族内部自洽，不是拼接产物。

### 切分

- IS 2020-01-01 ~ 2023-12-31 训练 / OOS 2024-01-01 ~ 2026-11-08 判定，与 E014/E017 同边界；**同一张考卷、同一批行**，不换行不删行。
- 标准化与特征选择只用 IS；本实验**不做特征选择**（列集在下方臂定义里跑前写死）。
- 臂定义（全部 `LGBMRegressor` 全默认，D044 冻结；方向由 `ŷ > log1p(vol_t)` 导出，固定 thr 0.5）：
  | 臂 | 列 |
  |---|---|
  | `m0` | E017 的 21 列原样（对照，必须逐位复现 E017 读数） |
  | `m_taker` | m0 + `taker_share`（主动买占比）、`taker_share_dev24`（对减过去 24 有效小时均值，shift 1）、`log1p(taker_buy_volume)`、`log1p(被动卖量)`、`trades_rel`（`log1p(count)` 减其 24 有效小时均值） |
  | `m_funding` | m0 + `funding_sum24`（过去 24h 已结算费率之和）、`funding_last`（最近一笔，带符号）、`funding_stale_h`（距今小时数） |
  | `m_both` | m0 + 上述 8 列 |
  **`trades_rel` 用相对值而非 `log1p(count)` 原值**是探针逼出来的：实测 BTC `count` 中位数 2020-01 = 10,037 笔/小时、2024-01 = 115,321 笔/小时，水平跨越 11 倍且随上市时间非平稳，原值进模型等于塞进一个「年代」变量。资金费率同理不用年化原值而用 24h 之和，因为区间不是恒定 8h（S3③）。
- 逐年（2024/2025/2026）分解、混淆矩阵、预测上涨份额 vs 基率，全部与 E017 同格式落盘。

### 成本与成交

- 不接实盘、无成交假设、无凭据。抓取量估算（由实测均值外推，E3）：69 币 × ~82 币-月 klines ≈ **172 MB** + 资金费率 ≈ **4.7 MB**，约 11,300 次 GET；8 并发 + 0.15s 间隔 ≈ 10~20 分钟。训练判定同 E017 量级（9.66 秒 × 4 臂）。上限 60 + 30 分钟。
- 并发披露：另一会话 E016 在读 `data/bars/`；本实验**只写新目录**，bars 与两个在案协议只读（S5）。

### 样本量与检验

- 判定行 1,686,328 / 69 币（与 E014、E017 同一批）；币簇 bootstrap（seed 20261110，10,000 次）只作报告项，**不进判据**——判据是跑前写死的点估计门槛。
- 多重比较披露：四臂对同一 m0 作差，主判据只认 `m_both`；`m_taker` / `m_funding` 属 C2 归因项，阳性/阴性结论不得由它们单独发起。

### 预期反例

- 若 `m_both` Δacc ≤ 0：新信息维度与同源加列一样无量，**关闭这条路**，此后「再提升」只剩数据质量（E008 线）与标签口径重定义两个方向。
- 若 Δacc > 0 但全部来自 `m_funding`、`m_taker` 为负：说明出力的是「市场拥挤度」而非「微观主动买卖」，表述须照此写，不得笼统写成「新维度有效」。
- 若 S1 全量对拍出现任何非零差值：判定行集与 bars 不再同源，本实验作废（不是重跑一遍就算过）。

## 2. 实现

依用户「冻协议 E019-1.0.0 开始全量抓取」与随后「好了吗，可以继续了吗」执行完链条：冻协议 → 抓取 + 校验 → S1 全量逐位对拍 → 训四臂 → S2/S3/S4/S5/S6 → 判定写回。

**抓取**（`code/fetch_newinfo.py`，8 并发 + 0.15 s 全局间隔 + 逐文件官方 `.CHECKSUM` sha256 校验，无凭据）：任务清单 = 4,361 个 klines 币-月 + 552 个当月 daily + 4,361 个 funding 币-月 = **9,274 次 GET**，落 `data/futures_flow/{1h,funding,raw}/`（69 + 69 个 parquet），**未写 `data/bars/`**（S5 的 bars sha256 清单跑后逐币 `unchanged = true`，`out/train/inventory_check.json`）。attempt1 记录 4,343/9,274 后中止，resume 补完 9,274/9,274（`complete = true`、`n_unrecorded_tasks = 0`、缓存命中不计重复下载）。

**四臂**（`code/train.py`）：`LGBMRegressor(random_state=0, n_jobs=4, verbosity=-1)` 全默认、`categorical_feature=["coin","hour","dow"]`、判定行 **1,686,328**、方向 `ŷ > col_lag0`、固定 thr 0.5，行集与评分函数由 `import baselines` / `import e017_train` 逐字继承（D042 第 5 条）。跑两遍（`out/train/`、`out/train_rerun/`）供 S6。

### 偏差（跑后照实记，不改判据一字）

1. **attempt1 传输中止 + 协议重冻 `E019-1.0.1`**：attempt1 跑到 4,343 个任务时 `GALAUSDT` daily 归档连续 4 次 `SSLEOFError` ⇒ 按协议「非 404 失败即中止、不写 parquet」停机（1.0.0 当时会保留半成品，这是 bug，见下）。因中止时**零判定数字**，D044 第 4 条的重冻窗口仍开着，故只重冻传输项（重试 6 次 / 退避 `[5,10,15,30,60,60]` / `cancel_futures` 取消逻辑），`arms`、`criteria`、`gates`、`inventory`、`manifest` 由 `build_protocol.py` 断言与 1.0.0 **逐字节相同**。留痕：`protocol.E019-1.0.0.json`（sha256 `d518ad72…`）、`out/fetch/attempt1_E019-1.0.0_aborted.json`、1.0.1 sha256 `ccbd69cc5f092b6b71d10c6a18db8700574d534445003c73ebb43c931a293a23`。
   取消逻辑错在 `fut.cancel()` 只对未开工的排队任务有效，`as_completed` 已经吐出的 future 必然已完成——改成 `break` + `pool.shutdown(wait=True, cancel_futures=True)`。
2. **抓取墙钟超预注册上限**：attempt1 2,917.6 s（48.6 min）+ resume 1,446.6 s（24.1 min）= **累计 4,364.2 s ≈ 72.7 min > 预注册的 60 min**（停止条件写的是「超限记录并停」）。两处口径本身也不一致：预注册 §成本写 60 min，协议 `fetch.wall_limit_seconds` 冻成了 5,400 s（90 min），单次运行未触协议门、累计触了文档门。判定链条没有因此被污染（数据 100% 落盘且过了 S1 硬门），但成本读数必须按 72.7 min 报，不得只报 24.1 min。
3. **S1 的「双向无孤儿」被 720 行打破，且断言按字面未通过**：`n_only_bars_total = 0` 而 `n_only_zip_total = 720`，全部来自 **STXUSDT 2024-09 整月在 `data/bars/` 里不存在**（bars 31,810 行 vs 归档 32,530 行）。三列 `max_abs_dev` 仍逐位 0.0。后果限定：判定行集由 bars 侧生成，bars 缺月 ⇒ 该月不可能进入考卷，实测判定行恰好 1,686,328（= E014/E017）。**但必须点明白这一步是放宽而非通过**：`code/check_join.py` 在写完 `out/join/` 全部证据后以非零退出（`AssertionError: S1 failed: archive rows absent from bars`），本实验仍继续训四臂，依据是「预注册 §预期反例的作废条款绑的是三列差值非零，本案 dev 全 0.0；孤儿方向在归档侧，构造上不可能进入 bars 生成的行集」——这条解释**由判定行数恰等 1,686,328 支撑**，不是由 S1 通过支撑。若用户认为字面条款优先，本案应判作废，重跑无意义（bars 缺月不会因重跑而补上），需要的是先补 STX 2024-09 的 bars 再判。证据 `out/join/join_check.csv`（`code == STXUSDT`：`n_bars = 31,810 / n_zip = 32,530 / n_only_zip = 720 / n_only_zip_months = ['2024-09']`）与 `out/join/join_summary.json`。
4. **首次四臂跑出的 m0 与 E017 差 −3.40pp，被 S2 硬门拒绝**（代码 bug，非协议问题）：`build_frames` 给评测帧写基线辅助列时用了 `ev["ma24"] = pairs["ma24"]`，而 `ma24` **同时是模型特征**（E017 `feature_frame` 里 = log1p(24 有效小时均量)，pair 侧 = 原始量纲均量），同名赋值把特征静默覆盖成错量纲 ⇒ 四臂全部在坏特征上打分（首跑 m0 0.63866）。定位证据 `out/diag/head2head.json`：两种写法逐列 sha1 对比，**train 帧 22 列 0 处不匹配**、**eval 帧只有 1 列不匹配，正是 `ma24`**——即训练矩阵逐位相同而预测不同。修法：辅助列改名 `pair_vol` / `pair_ma24`，并加「辅助列名不得与臂列名相交」断言（`code/train.py:200`）。此时判定数字已存在，D044 第 4 条的重冻窗口关闭，故**协议、判据、列集、行集一字未动，只改代码**；改后 18 项 S2 检查全部 `bitwise_equal = true`。首跑产物留在 `out/train_archive_clobma24bug/`，**不作任何结果引用**。

## 3. 结果（E1）

判定行 1,686,328 / 69 币，与 E014、E017 同一张考卷。产物：`exp/E019-new-info-dimension/out/{fetch,join,train,s3_no_lookahead.json,s6_determinism.json}`。

### 3.1 四臂总表（逐币等权为主口径）

| 臂 | 列数 | eq_acc | Δacc vs m0 | eq_bacc | Δbacc | pooled_acc | Δpooled | eq_precision | eq_recall | eq_mae_log1p | 中位 acc | min | max | ≥0.70 币数 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `m0`（= E017 m1） | 21 | **0.672704** | — | 0.674900 | — | 0.672807 | — | 0.63908 | 0.71660 | 0.35737 | 0.67512 | 0.59155 | 0.68750 | 0 |
| `m_taker` | 26 | 0.673950 | **+0.1246pp** | 0.676013 | +0.1113pp | 0.674051 | +0.1244pp | 0.64098 | 0.71520 | 0.35657 | 0.67647 | 0.59507 | 0.68966 | 0 |
| `m_funding` | 24 | 0.673022 | **+0.0318pp** | 0.675119 | +0.0219pp | 0.673148 | +0.0341pp | 0.63975 | 0.71515 | 0.35735 | 0.67527 | 0.59499 | 0.68719 | 0 |
| `m_both` | 29 | **0.673964** | **+0.1260pp** | 0.675963 | +0.1064pp | 0.674072 | +0.1265pp | 0.64131 | 0.71401 | 0.35656 | 0.67639 | 0.59247 | 0.69022 | 0 |

- 参照（同卷重算，非引用）：E014 a5 `ma24` 回归 0.62382（`baselines_recomputed.csv`，与 E014 冻结产物差 ≤1e-9）；E017 m1 0.672704。四臂预测平局数 0。
- 逐年 eq_acc 分解：`m0` 2024/2025/2026 = 0.67145 / 0.67174 / 0.67533；`m_both` = 0.67273 / 0.67314 / 0.67591 ⇒ Δ = **+0.128 / +0.140 / +0.058pp**；`m_taker` Δ = +0.139 / +0.123 / +0.071pp；`m_funding` Δ = +0.049 / +0.044 / +0.003pp。增益三年同号但每年都在 0.15pp 以内。

### 3.2 C1 主判定

Δacc(`m_both`) = **+0.00126**（门槛 +0.010）⇒ **`marginal_new_info`**：方向为正、pooled 与 bacc 同向（+0.1265pp / +0.1064pp）、MAE 也降（−0.00080），但只有门槛的 **1/8**。70% 目标未达（等权 0.6740、最好单币 0.69022、0/69 达标）。

币簇 bootstrap（seed 20261110、10,000 次、69 币，**只作报告项**）：

| 臂 | 均值 Δ | 中位 Δ | 95% CI | 为正币数 |
|---|---|---|---|---|
| `m_both` | +0.1260pp | +0.1440pp | [+0.0924pp, +0.1601pp] | 55/69 |
| `m_taker` | +0.1246pp | +0.1318pp | [+0.0944pp, +0.1548pp] | 54/69 |
| `m_funding` | +0.0318pp | +0.0240pp | [+0.0069pp, +0.0572pp] | 43/69 |

即：增量为正且稳（CI 不跨 0），但**量级本身比门槛小一个数量**——把抽样噪声排除干净也够不着 +1.0pp。

### 3.3 C2 分维度归因

`|eq(m_both) − eq(m_taker)|` = **1.42e-5**（< 0.3pp 规则线）⇒ 依预注册表述写：**增量归属 taker，资金费率无额外贡献**（不得写成「两者都有贡献」）。m_both 的 gain 份额：基列组 97.216%、taker 组 **2.780%**、funding 组 **0.00375%**。逐列（m_both）：

| 新列 | gain 份额 | split 数 |
|---|---|---|
| `log1p_taker_buy_vol` | **2.7101%** | 73 |
| `log1p_passive_sell_vol` | 0.0575% | 34 |
| `trades_rel24` | 0.0116% | 160 |
| `funding_sum24` | 0.0015% | 27 |
| `funding_stale_h` | 0.0013% | 27 |
| `funding_last` | 0.0009% | 22 |
| `taker_share_dev24` | 0.0006% | 18 |
| `taker_share` | 0.0003% | 9 |

8 列里 **1 列**（`log1p_taker_buy_vol`）拿走 2.71%，其余 7 列合计 0.075%。全局排序上它已挤进第 3 位（`col_lag0` 72.57%、`ewma94` 19.83%、**它 2.71%**、`ma6` 2.12%）。

### 3.4 硬自检读数

- **S1**：69/69 币 `max_abs_dev_volume = max_abs_dev_close = max_abs_dev_amount = 0.0`（**作废条款绑的这条全过**）；「双向无孤儿」这一子条**未通过**——`n_only_bars_total = 0`、`n_only_zip_total = 720`（STXUSDT 2024-09），`check_join.py` 按字面断言以非零退出，详见 §2 偏差 3 的放宽理由。`taker_share` 越界 0 次；有效 bar（`volume>0`）共 3,136,850 根中 `count ≤ 0` **0 根**、`taker_buy_volume ≤ 0` **4 根**（死量 bar 31,662 根不进特征）。
- **S2**：18 项检查（m0 的 14 个逐币指标 + 3 项汇总 + a5 对 E014 产物）**全部 `bitwise_equal = true`**，`out/train/s2_checks.json`。
- **S3**：8/8 组通过（BTC/ETH/JTO/SOL × volume/amount）；每组含截断×3、未来扰动、以及 **T3 t+1h 结算放大不改 t 行**；`out/s3_no_lookahead.json`。
- **S4**（分母 = 1,686,328 判定行）：8 个新列**判定行 NaN 单元格数为 0**（任何 taker/funding 列为 NaN 的行 = 0 行）——即新信息在考卷上是满覆盖的，无量不是缺数据造成的。`funding_sum24 == 0` 21,117 行 = **1.2522%**；距上次结算小时数 中位 3h / p95 7h / 最大 **198h**；判定行上完全无资金费率的币 0 个（抓取清单里 50 个 funding 币-月 404，落在 bars 窗口之外或退市死段）。
- **S5**：69 个 bars 文件 + E014/E017 协议 sha256 跑后逐位不变，`out/train/inventory_check.json` 全 `unchanged: true`。
- **S6**：5 个文件两遍 sha1 逐位一致（含 protocol.json 的两次运行自记 sha256 交叉核对），`out/s6_determinism.json`。
- **墙钟**：训练判定 22.25 s / 重跑 23.76 s（上限 1,800 s，未超）；抓取累计 72.7 min（超 60 min，见偏差 2）。

## 4. 结论

**一句话：换成真正不同源的两类信息，实测只买到 +0.126pp（门槛 +1.0pp 的 1/8），而且这 0.126pp 全部出自「主动买量」那一列，资金费率没有额外贡献（+0.032pp，gain 0.004%）。判定 = `marginal_new_info`，70% 目标仍差 2.6pp。**

三点必须分开写：

1. **增量是真的，但小一个数量级。** Δacc、Δpooled、Δbacc、ΔMAE 四项同号，币簇 bootstrap CI [+0.092pp, +0.160pp] 不跨 0，55/69 币为正，且 2024/2025/2026 三年同号——这不是抽样噪声，也不是某一年撑起来的。但它离 +1.0pp 门槛差 8 倍，**即使把噪声完全排除也够不着**，所以不构成「模型层级」的改变。
2. **资金费率这条信息在本卷上接近无量，是阴性而不是"没测到"。** 判定行上 8 个新列 **NaN 单元格为 0**（满覆盖），`funding_sum24` 只有 1.25% 的行为 0，距上次结算中位 3h——数据到位、新鲜、无缺；模型仍只给 0.004% gain。同时 S3/T3 已证明「t+1h 那一笔结算一律不用」的保守口径成立，所以不能把无量归给前视处理。表述照预注册 C2 写死：**增量归属 taker，资金费率无额外贡献**。
3. **taker 那一列的功劳里混着同源信息，本设计无法分离——这是本结论的限定项，不是补救项。** `log1p_taker_buy_vol = log(vol_t) + log(主动买占比)`，而 `vol_t` 的量级特征侧本来就有（`col_lag0` 就是当根 log1p 量）。它拿到 2.71% gain 而**纯占比**列 `taker_share` / `taker_share_dev24` 只拿 0.0009%，提示「出力的是量级 × 占比的乘性写法，不是占比本身」，但要把两者彻底切开需要再加一臂（例如只喂占比、不喂绝对量级），**预注册停止条件禁止加臂**，故此处只写 uncertainty，不追跑。

按预注册「预期反例」与停止条件：Δacc > 0 但远低门槛 ⇒ 不判死路、也不判有效，**这条路按「已试、无可用量级」关闭**；不扫参、不换列集、不加第二个标签口径。此后「把成交量模型推过 70%」只剩两个方向：**数据质量**（E008 线，本仓实测死量 bar 31,662 根仍在卷内）与**标签口径重定义**（现在的标签是"下一小时总量涨/跌"，70% 可能是这个口径本身的天花板——E014 已证基率 47.4%、最强零训练 62.4%）。

## 5. 复现

```bash
PY=.venv/bin/python   # 3.12.16 / pandas 3.1.1 / numpy 2.5.4 / lightgbm 4.7.0；系统 python3=3.14 无 pyarrow，读 parquet 必炸

# 0) 探针（只读 bars，几百 KB，可重复）
$PY exp/E019-new-info-dimension/code/probe_newinfo.py

# 1) 冻协议（含 9,274 任务清单 + bars sha256 钉版；重冻须零判定数字）
$PY exp/E019-new-info-dimension/code/build_protocol.py

# 2) 全量抓取（≈72.7 min 累计、146 MB、写 data/futures_flow/，可断点续抓）
$PY exp/E019-new-info-dimension/code/fetch_newinfo.py

# 3) S1 全量逐位对拍 + S4 覆盖前置（硬门，失败也先落盘证据）
$PY exp/E019-new-info-dimension/code/check_join.py   # 以非零退出：归档侧 720 行孤儿（§2 偏差 3），out/join/ 全部证据在此之前已写完

# 4) 四臂训练与判定（≈22 s；S2/S5 在此断言）
$PY exp/E019-new-info-dimension/code/train.py
$PY exp/E019-new-info-dimension/code/train.py --out exp/E019-new-info-dimension/out/train_rerun   # S6 第二遍

# 5) S3 防前视 8 组 / S6 两遍 sha1
$PY exp/E019-new-info-dimension/code/test_no_lookahead_newinfo.py
$PY exp/E019-new-info-dimension/code/check_determinism.py

# 6) 定位 §2 偏差 4 的量具（train 帧逐列 sha1 对比，产出 out/diag/head2head.json）
$PY exp/E019-new-info-dimension/code/diag_head2head.py
$PY exp/E019-new-info-dimension/code/diag_m0.py            # 单币帧 + 两次拟合逐位对照
$PY exp/E019-new-info-dimension/code/diag_all_coins.py     # 69 币 train 掩码逐列比对
$PY exp/E019-new-info-dimension/code/diag_pooled.py        # 池化顺序/切分四种写法
```

产物不入库（`.state/`、`data/`、`exp/**/out/` 规则）；重跑第 2 步需联网访问 `data.binance.vision`，其余全只读本地。

## 6. 变更记录

- 2026-11-10：依用户「同意，请继续」（上一轮报出的实测可行性）建立预注册并登记（台账 E019 行 + 判定日志 **#65**）。**取号**：`ls docs/30_实验/`（最新 E017）+ `ls exp/` + 台账行 + 全库 grep 四处核过——E015 按路线图留给分层回退查表、**E018 已被并发会话预告给「EWMA/GARCH 预测值作特征」**（`docs/30_实验/E016-HAR窗口尺度规则与快档补全.md:165`），故本实验取 E019；决策号未消费（下一空闲 D047）。**本次产出 = 探针脚本 + 6 件探针产物 + 本预注册**，全量抓取未开始、四臂未训练、`data/` 未写入、协议未冻、`data/bars/` 与 E014/E017 协议只读未动。**探针改变设计的三处实测**（按 D044 第 1 条由脚本落盘，不从对话抄）：① 资金费率区间非恒定 8h（TON 同月 4h/8h 并存）⇒ 特征改用「过去 24h 之和」并禁硬编码；② `count` 水平非平稳（BTC 10,037 → 115,321 笔/小时）⇒ 只进相对值；③ bars 与归档三列逐位相同（`max_abs_dev = 0.0`）⇒ taker 列的拼接不需要任何对齐假设，但也因此 S1 的全量对拍是硬门而非形式。未提交。
- 2026-11-11：跑完全链条并**判定 `marginal_new_info`**（Δacc +0.126pp，门槛 +1.0pp；C2 归因 taker、资金费率无贡献）。**四条偏差全部照实写入 §2**：① attempt1 于 4,343/9,274 处因 `SSLEOFError` 中止 ⇒ 在「零判定数字」窗口内重冻 `E019-1.0.1`（只改重试/退避/取消，arms-criteria-inventory-manifest 逐字节断言不变，1.0.0 与中止记录留盘）；② 抓取累计墙钟 72.7 min **超预注册 60 min 上限**（且协议 `wall_limit_seconds` 冻成 5,400 s 与文档口径不一致，一并记）；③ S1 归档侧孤儿 720 行 = **STXUSDT 2024-09 整月在 bars 里不存在**，三列 dev 仍 0.0、判定行恰 1,686,328 ⇒ 只推翻覆盖表述，不推翻判定；④ **首次四臂 m0 = 0.63866 与 E017 差 −3.40pp，被 S2 硬门拒绝**——根因 `build_frames` 用 `ev["ma24"] = pairs["ma24"]`（a5 基线的原始量纲均值）**覆盖了同名模型特征 `ma24`**（E017 里是 log1p 后的均值），定位靠 `out/diag/head2head.json` 的逐列 sha1：train 帧 22 列 0 处不匹配、eval 帧唯一不匹配列即 `ma24`；改法 = 辅助列改名 `pair_vol/pair_ma24` + 名字不相交断言（`code/train.py:200`），**协议与判据一字未动**（判定数字已存在，D044 第 4 条窗口已关），改后 18 项 S2 全 PASS，坏产物移入 `out/train_archive_clobma24bug/` 不引用。**本次产出 = 9,274 次抓取 + 138 个 parquet + S1…S6 全套产物 + 判定写回 + D047 + 教训一节**；`data/bars/`、E014/E017 协议与产物、`vendor/` 只读未动，未提交。
