"""校验内置 framework 并安装本地数据源适配器。

用法:
    python scripts/setup_framework.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
VENDOR = ROOT / "vendor" / "LocalStore.py"


def main():
    if not (FRAMEWORK / "Chan.py").exists():
        raise SystemExit(f"{FRAMEWORK} 不完整：缺少 Chan.py")

    dst = FRAMEWORK / "DataAPI" / "LocalStore.py"
    shutil.copyfile(VENDOR, dst)
    print(f"copied {VENDOR} -> {dst}")
    print("done. 下一步: python scripts/fetch.py")


if __name__ == "__main__":
    sys.exit(main())
