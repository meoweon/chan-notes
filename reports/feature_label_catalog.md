# 特征、标签与训练逻辑（v1）

> 数据：`data/events/events_v2.parquet`，17,192 行 × 188 列
> 其中 **131 列特征**、**32 列标签**、**25 列元信息**
> 生成日期：2026-11-01

## 一、特征目录（131 列）

所有特征在事件 bar `t` 现场计算，只读取 `≤ t` 的数据；列名固定，schema 落盘校验，出现增删列会直接报错。

### 1.1 价格行为 · 单根 K 线（pa_，54 列）

| 特征 | 含义 |
|---|---|
| `pa_body_ratio` | 实体 / 振幅 |
| `pa_upper_shadow` / `pa_lower_shadow` | 上 / 下影线占振幅比 |
| `pa_close_pos` | 收盘价在振幅中的相对位置（0 低、1 高） |
| `pa_is_bull` | 阳线 +1 / 阴线 -1 / 十字 0 |
| `pa_gap_up` / `pa_gap_dn` | 相对前一根的跳空 |
| `pa_inside` / `pa_outside` | 内包 / 外包 |
| `pa_hh` / `pa_ll` / `pa_hl` / `pa_lh` | 与前 5 根的高低点关系（更高高/更低低/更高低/更低高） |
| `pa_streak` | 连续同向 bar 数（带方向，上限 10） |
| `pa_ret1` / `pa_ret5` / `pa_ret20` / `pa_ret60` | 1/5/20/60 bar 收益率 |
| `pa_range_expansion` | 本根振幅 / 前一根振幅 |
| `pa_ft_dir_match` | 与前一根同向 |
| `pa_ft_break` | 顺向突破前一根高/低 |
| `pa_ft_body` | 同向时取实体比，反向取负 |
| `pa_ft_strength` | 跟随强度 = 0.5×同向 + 0.3×突破 + 0.2×实体 |
| `pa_range_pos20` | 收盘在 20 bar 区间的位置 |
| `pa_dist_hh20` / `pa_dist_ll20` | 距 20 bar 高 / 低的百分比 |
| `pa_breakout_up` / `pa_breakout_dn` | 收盘突破 20 bar 高 / 低 |
| `pa_fail_break_up` / `pa_fail_break_dn` | 假突破（影线越界、收盘回到区间） |
| `pa_range_width20` | 20 bar 区间宽度 / 价格 |
| `pa_slope20_norm` | 20 bar 线性回归斜率 / 价格 |
| `pa_r2_20` | 回归 R²（通道质量） |
| `pa_trend_dir` | R²≥0.55 时的趋势方向 |
| `pa_boll_pos` / `pa_boll_width` | 收盘在布林带中的位置 / 带宽 |
| `pa_boll_break_up` / `pa_boll_break_dn` | 突破布林上 / 下轨 |
| `pa_ma_align` | 5>10>20 多头排列 +1，反之 -1 |
| `pa_price_vs_ma20` / `pa_ma5_vs_ma20` | 价格 / MA5 相对 MA20 |
| `pa_bar_range_atr` | 本根振幅 / ATR14 |
| `pa_atr_pct` | ATR14 / 价格 |
| `pa_std_pct` | 20 bar 标准差 / 价格 |
| `pa_macd_dif` / `pa_macd_dea` / `pa_macd_bar` | MACD 三线（除以价格归一） |
| `pa_macd_cross` | 金叉 +1 / 死叉 -1 |
| `pa_rsi` / `pa_rsi_ob` / `pa_rsi_os` | RSI14 / 超买 / 超卖 |
| `pa_vol_ratio5` / `pa_vol_ratio20` | 量比（对 5 / 20 bar 均量） |
| `pa_vol_on_break` | 突破日的量比（未突破为 0） |

### 1.2 价格行为 · 形态（pt_，15 列）

| 特征 | 含义 |
|---|---|
| `pt_n_swing_high` / `pt_n_swing_low` | 过去 120 bar 已确认摆动点数量 |
| `pt_double_top` / `pt_double_bottom` | 双顶 / 双底（两摆动点价差 ≤0.8ATR、间隔 ≥5 bar、回撤 ≥1ATR） |
| `pt_dt_neck_dist` / `pt_db_neck_dist` | 现价距双顶/双底颈线的百分比 |
| `pt_wedge_up` / `pt_wedge_dn` | 楔形（三次递进推动且后一段力度衰减 20%） |
| `pt_always_in` | 连续 5 根在 MA20 同侧（+1/-1/0） |
| `pt_bo_pb_long` / `pt_bo_pb_short` | 突破后回踩不破（Brooks 的 BO-PB） |
| `pt_mm_room_up` / `pt_mm_room_dn` | 测量移动目标空间（最近摆动区间高度） |
| `pt_spike` | 3 bar 振幅 > 2.5×ATR |
| `pt_tight_range` | 20 bar 区间 < 3×ATR（窄幅震荡） |

### 1.3 缠论结构（ch_，54 列）

| 特征 | 含义 |
|---|---|
| `ch_n_bi` / `ch_n_sure_bi` | 笔总数 / 已确认笔数 |
| `ch_bi_dir` / `ch_bi_sure` | 当前笔方向 / 是否确认 |
| `ch_bi_klc_cnt` / `ch_bi_klu_cnt` | 当前笔包含的合并 K 线 / 原始 K 线数 |
| `ch_bi_amp` / `ch_bi_amp_prev` / `ch_bi_amp_ratio` | 当前笔幅度 / 上一笔幅度 / 比值 |
| `ch_bi_slope` | 笔幅度 / 笔 K 线数 |
| `ch_bi_pos` | 当前 K 线在笔内的位置（0~1） |
| `ch_bi_idx` | 笔序号 |
| `ch_bi_dir_streak` | 连续同向笔数（**实测恒为 1，退化**） |
| `ch_bi_in_seg_pos` | 当前笔在所属线段中的位置 |
| `ch_seg_dir` / `ch_seg_sure` | 当前线段方向 / 是否确认（**sure 恒为 0，退化**） |
| `ch_seg_bi_cnt` / `ch_seg_klu_cnt` / `ch_seg_amp` | 线段笔数 / K 线数 / 幅度 |
| `ch_seg_n_zs` / `ch_seg_has_trendline` | 线段内中枢数 / 是否有趋势线 |
| `ch_n_zs` | 中枢总数 |
| `ch_zs_high` / `ch_zs_low` / `ch_zs_width` | 当前中枢上下沿 / 宽度 |
| `ch_zs_bi_cnt` / `ch_zs_is_one_bi` | 中枢笔数 / 是否单笔中枢 |
| `ch_zs_pos` / `ch_in_zs` | 收盘在中枢中的相对位置 / 是否在中枢内 |
| `ch_zs_dist_high` / `ch_zs_dist_low` | 距中枢上 / 下沿百分比 |
| `ch_zs_peak_width` | 中枢峰值区间宽度 |
| `ch_price_vs_zs_mid` | 收盘相对中枢中值 |
| `ch_bi_leaves_zs` | 当前笔是否已离开中枢 |
| `ch_bars_since_zs_end` | 中枢结束以来 bar 数 |
| `ch_bsp_1` `ch_bsp_1p` `ch_bsp_2` `ch_bsp_2s` `ch_bsp_3a` `ch_bsp_3b` | 买点类型 one-hot（同一次可多类） |
| `ch_bsp_is_buy` | 方向 |
| `ch_bsp_div` | 背驰度 `divergence_rate` |
| `ch_bsp_zs_cnt` | 形成该 BSP 时的中枢数 |
| `ch_bsp_bi_amp` | BSP 所在笔幅度 |
| `ch_bsp2_retrace_rate` / `ch_bsp2_break_bi_amp` / `ch_bsp2_bi_amp` | 二类买点回撤率 / 破坏笔幅度 / 本笔幅度 |
| `ch_bsp2s_retrace_rate` / `ch_bsp2s_break_bi_amp` / `ch_bsp2s_bi_amp` / `ch_bsp2s_lv` | 二类严格版同上 + 级别偏移 |
| `ch_bsp3_zs_height` / `ch_bsp3_bi_amp` | 三类买点中枢高度 / 笔幅度 |

> 空值特征（std=0）：`ch_bi_dir_streak`（恒 1）、`ch_seg_sure`（恒 0）、`ch_zs_is_one_bi`（恒 0）。低覆盖（<10% 非零）：`pa_inside`、`pt_bo_pb_*`、`pa_gap_*`、`ch_bsp_3a/3b`。

### 1.4 市场环境（mkt_，8 列）

| 特征 | 含义 |
|---|---|
| `mkt_ret5` / `mkt_ret20` | 等权市场指数 5 / 20 bar 收益 |
| `mkt_above_ma20` / `mkt_above_ma60` | 指数在均线上方 |
| `mkt_ma20_slope` | 指数 MA20 斜率 |
| `mkt_atr_pct` | 指数 ATR / 价格 |
| `mkt_rsi` | 指数 RSI14 |
| `mkt_range_pos20` | 指数在 20 bar 区间的位置 |

> 市场指数 = 全部样本股 30m 收盘价归一化后取中位数，等权合成（`scripts/build_market.py`）。

### 1.5 Brooks 结构（23 列，2026-11-01 新增）

移植自 [RaiderYi/price-action-lab](https://github.com/RaiderYi/price-action-lab)（MIT），
把其因果实现改写为 numpy 纯函数，摆动点保留"右侧 3 根确认"的约束。

| 特征 | 含义 |
|---|---|
| `sw_n` / `sw_bull_ratio` / `sw_label` | 已确认摆动点数量 / HH+HL 占比 / 结构标签（±1） |
| `sw_dist_h` / `sw_dist_l` / `sw_bars_since` | 距最近摆动高 / 低的百分比 / 距最近摆动点的 bar 数 |
| `rg_label` / `rg_quality` | 环境三态（趋势 ±1 / 区间 0） / 判定质量分 |
| `rg_bull_ratio` / `rg_ma_bull` / `rg_range_atr` | 结构占比 / MA20>MA60 / 20 bar 区间宽度（ATR 倍数） |
| `bc_h` / `bc_l` / `bc_signal_age` / `bc_side` | H1/H2… 与 L1/L2… 的累计计数 / 最后一次计数距今 bar 数 / 计数方向 |
| `bsw_level` / `bsw_big` / `bsw_extreme` / `bsw_crossed` | 突破强度 0–3（大棒+极端收盘）/ 大棒 / 极端收盘 / 突破方向 |
| `ff_flag` | final flag（5 根紧凑且小实体） |
| `rf_long` / `rf_short` | 区间边缘反向（宽度 ≥2.5ATR 且位置 ≤0.25 / ≥0.75） |
| `wg_quality` | 三推楔形的推力衰减度（负值=衰减） |

> 因果性：摆动点必须右侧 3 根确认后才输出；`bc_*` 以"最近一个已确认摆动低点"为锚点；
> 突破强度只看突破棒本身，跟随棒不参与。截断重跑检验通过（0 mismatch）。
> 实测结论：这 23 列在 6,169 个确认买点上**没有带来 AUC 增量**（见 `feature_effectiveness.md`）。

## 二、标签目录（32 列）

### 2.1 入场时点（元信息，25 列）

| 列 | 含义 |
|---|---|
| `code` `freq` `bar_idx` `event_time` | 股票、周期、决策 bar、决策时间 |
| `bsp_bi_idx` `bsp_klu_idx` `bsp_type` `is_buy` | BSP 锚定的笔、K 线、类型、方向 |
| `delay_bars` | 决策 bar 与 BSP 锚定 K 线的距离（BSP 滞后确认的 bar 数） |
| `entry_idx` `entry_time` `entry_price` | 信号口径入场：次 bar 开盘 |
| `atr` | 事件 bar 的 ATR14（用于止损） |
| `conf_ok` `conf_wait` `conf_entry_idx` `conf_entry_time` `conf_entry_price` `conf_atr` | 确认口径入场：N bar 内突破信号 bar 高/低点后，次 bar 开盘 |

### 2.2 收益标签（信号口径，20 列）

对 H ∈ {8, 24, 48}（约 1 / 3 / 6 个交易日）：

| 列 | 定义 |
|---|---|
| `ret_H` | `close[entry+H-1] / entry - 1 - 0.0013`（多头原始收益） |
| `dir_ret_H` | 买点取 `ret_H`，卖点取 `-ret_H`（信号一致性收益） |
| `mfe_H` | 持有期内最大有利偏移（顺信号方向） |
| `mae_H` | 持有期内最大不利偏移 |
| `stop_H` | 持有期内是否触及 `entry ∓ 2×ATR` |

### 2.3 策略一致性标签（信号口径，4 列）

| 列 | 定义 |
|---|---|
| `pol_ret` | 按真实规则模拟：H=24、2×ATR 止损、先到先出 |
| `pol_dir_ret` | 上述结果按信号方向取号 |
| `pol_bars` | 实际持有 bar 数 |
| `pol_stop` | 是否止损离场 |

### 2.4 确认口径标签（8 + 20 列）

`c_pol_*`（4 列）与 `c_ret_H / c_dir_ret_H / c_mfe_H / c_mae_H / c_stop_H`（20 列）与上面定义完全相同，唯一区别是入场时点换成 `conf_entry_*`。未确认的事件这些列为 `NaN`。

## 三、训练逻辑

### 3.1 样本生成

1. `CChan.step_load()` 逐 bar 回放，BSP 首次出现在 store 时记录事件（因果，不做 batch 终态筛选）。
2. 事件 bar `t` 上计算 131 个特征；`t` 之后的数据只用于标签。
3. 只保留 `t ≥ warmup(300)` 的事件；同一 `(bi_idx, 类型)` 只记一次。

### 3.2 当前默认训练口径

| 项 | 取值 |
|---|---|
| 子集 | `is_buy == 1 且 conf_ok == 1`（确认入场，3,031 条 / 17,192） |
| 标签 | `c_pol_dir_ret > thr`；thr=0 用于"是否盈利"，thr=0.02 用于"是否大波" |
| 模型 | XGBoost `binary:logistic`，max_depth=3，eta=0.05，120 轮，subsample=0.8，colsample=0.8，min_child_weight=20 |
| 特征筛选 | 训练集内单值（nunique≤1）的列剔除，其余全部进入 |
| 验证 | 月度 walk-forward：训练集 = `event_time ≤ 月初-10 天`（标签成熟期），测试集 = 当月 |
| 最小训练量 | 600 条（确认子集）/ 1,500 条（全样本） |

### 3.3 评估

- **AUC**（Mann-Whitney 形式，不依赖 sklearn）
- **分位均值**：OOF 分数前 20% / 后 20% 的 `c_pol_dir_ret` 均值差
- **组合回测**：逐 bar 盯市，单笔 10% 仓位、最多 10 仓、1.5×ATR 止损、48 bar 时间退出、反向 BSP 离场、13bp 往返成本
- **稳定性检验**：前 20% 与前 10% 的期望是否单调（当前不单调 → 判定为噪声）

### 3.4 已跑的四种组合

| 实验 | 子集 | 标签 | 结果 |
|---|---|---|---|
| WF-1 | 全部事件 | `pol_dir_ret > 0` | 买点 AUC 0.487 |
| WF-2 | 买点 | `pol_dir_ret > 0` | 买点 AUC 0.487 |
| WF-3 | 确认买点 | `c_pol_dir_ret > 0` | AUC 0.502，前 20% +5.5bp / 后 20% -8.7bp |
| WF-4 | 确认买点 | `c_pol_dir_ret > 2%` | AUC 0.575（T2 0.605），前 20% -0.8bp / 后 20% -15.4bp |

### 3.5 尚未纳入的环节

- 横截面排序（同一时刻在全部候选中排序，而不是绝对概率）
- 仓位模型（把分数映射为仓位，而不是二分类过滤）
- 市场状态交互（只在特定 regime 内训练/交易）
- 参数搜索（`bi_strict` / `divergence_rate` 等保持官方默认，旧项目已验证参数无关性）
