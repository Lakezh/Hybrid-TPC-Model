#!/usr/bin/env python3
"""
依赖包安装脚本
运行方式：python install_packages.py
"""

import subprocess
import sys
import os

# Anaconda 在 Windows 上有时 SSL DLL 不在 PATH 中，需要手动加入
# 如果你的 Anaconda 装在其他位置，修改下面的路径
_anaconda_lib = r"D:\anaconda\Library\bin"
if os.path.isdir(_anaconda_lib) and _anaconda_lib not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _anaconda_lib + ";" + os.environ.get("PATH", "")

def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[警告] 上条命令返回非零退出码 {result.returncode}，请检查输出")
    return result.returncode

pip = [sys.executable, "-m", "pip"]

# ── 基础科学计算包 ────────────────────────────────────────────
run(pip + ["install", "-U", "numpy", "pandas", "matplotlib", "scikit-learn"])

# ── PyTorch ──────────────────────────────────────────────────
# 当前环境：NVIDIA CUDA 13.0 驱动，使用 cu128（向下兼容）

# CUDA 12.8 版（有 GPU 时使用）：
run(pip + ["install", "-U", "torch", "torchvision", "torchaudio",
           "--index-url", "https://download.pytorch.org/whl/cu128"])

# CPU 版（无 GPU 时取消注释，注释掉上面 CUDA 那行）：
# run(pip + ["install", "-U", "torch", "torchvision", "torchaudio"])

# ── ODE 求解器（可选，安装失败时代码自动 fallback 到内置 RK4）──
run(pip + ["install", "-U", "torchdiffeq"])

# ── XGBoost（多模型对比实验用）───────────────────────────────
run(pip + ["install", "-U", "xgboost"])

# ── Pillow（修复 Anaconda 环境下 matplotlib DLL 冲突）────────
run(pip + ["install", "-U", "pillow>=10.0"])

# ── 验证安装 ─────────────────────────────────────────────────
print("\n" + "=" * 50)
print("验证安装结果：")
print("=" * 50)

checks = [
    ("numpy",        "import numpy as np; print('numpy', np.__version__)"),
    ("pandas",       "import pandas as pd; print('pandas', pd.__version__)"),
    ("matplotlib",   "import matplotlib; print('matplotlib', matplotlib.__version__)"),
    ("scikit-learn", "import sklearn; print('scikit-learn', sklearn.__version__)"),
    ("torch",        "import torch; print('torch', torch.__version__); "
                     "print('CUDA available:', torch.cuda.is_available())"),
    ("torchdiffeq",  "from torchdiffeq import odeint; print('torchdiffeq OK')"),
    ("xgboost",      "import xgboost; print('xgboost', xgboost.__version__)"),
]

for name, code in checks:
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  [OK] {result.stdout.strip()}")
    else:
        print(f"  [FAIL] {name}: {result.stderr.strip()[:80]}")

print("\n安装完成。")