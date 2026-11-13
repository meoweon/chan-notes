"""项目路径与全局配置。

仓库根目录默认取本文件上两级，可用环境变量 CHANQ_ROOT 覆盖。
"""
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("CHANQ_ROOT", Path(__file__).resolve().parents[1]))
FRAMEWORK = ROOT / "framework"
DATA = ROOT / "data"
BARS = DATA / "bars"
EVENTS = DATA / "events"
MODELS = DATA / "models"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

for p in (BARS, EVENTS, MODELS, REPORTS, LOGS):
    p.mkdir(parents=True, exist_ok=True)

# 供 framework/DataAPI/LocalStore.py 读取（该文件在引擎进程里独立导入）
os.environ.setdefault("CHAN_NOTES_BARS", str(BARS))

if str(FRAMEWORK) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK))

# framework 参数基线：贴默认配置，不再使用旧项目的“超宽松版”
def chan_config(trigger_step: bool = True, bs_type: str = "1,1p,2,2s,3a,3b"):
    from ChanConfig import CChanConfig
    return CChanConfig({
        "trigger_step": trigger_step,
        "print_warning": False,
        "print_err_time": False,
        "bi_strict": True,
        "bi_fx_check": "strict",
        "divergence_rate": 0.9,
        "min_zs_cnt": 1,
        "bs1_peak": True,
        "macd_algo": "peak",
        "bs_type": bs_type,
        "zs_algo": "normal",
        "bsp2s_follow_2": False,
        "max_bs2_rate": 0.9999,
        "kl_data_check": False,
        "cal_rsi": True,  # dynamics 特征的 RSI 笔度量需要 klu.rsi（不影响 BSP 判定）
    })
