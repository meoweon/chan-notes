"""台账 #37 口径修复后的全量重算（后台串行，子进程隔离执行）。

顺序：摆动点电池 → BTC 电池×2 → 摆动点安慰剂 → 3 组 WF 重跑（purge=28d、独立验证集、月界修复）。
日志重定向到 reports/redo_fix_log.txt。
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["CHAN_NOTES_BARS"] = str(ROOT / "data" / "bars_crypto")  # BTC 数据根，子进程继承
os.environ["PYTHONIOENCODING"] = "utf-8"
PY = sys.executable


def run_script(rel, argv=None):
    print(f"\n{'=' * 20} {rel} {'=' * 20}", flush=True)
    cmd = [PY, "-X", "utf8", str(ROOT / rel)] + (argv or [])
    rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
    print(f"[exit {rc}] {rel}", flush=True)
    return rc


rcs = []
rcs.append(run_script("scripts/swing_battery.py"))
rcs.append(run_script("scripts/btc_battery.py"))
rcs.append(run_script("scripts/btc_battery_5m.py"))
rcs.append(run_script("scripts/swing_placebo.py"))

rcs.append(run_script("scripts/train_wf.py", ["events_v7s", "wf_surv_fix", "survive_144", "all", "1500", "0", "12"]))
rcs.append(run_script("scripts/train_wf.py", ["events_v7s", "wf_surv_cb_fix", "survive_144", "conf_buy", "500", "0", "12"]))
rcs.append(run_script("scripts/train_wf.py", ["events_v8", "wf_v6_dy_fix", "pol144_ret", "conf_buy", "500", "0", "12"]))
print(f"\nALL DONE exits={rcs}")
