# Group IV — PINN Thermal Performance Curve Model

> 基于物理信息神经网络（PINN）的微生物热性能曲线预测模型（第四组）

---

## 项目简介

本项目使用**物理信息神经网络（PINN）**结合**通用热性能曲线（UTPC）**约束，对微生物（细菌、古菌）的生长速率–温度曲线进行预测。

输入：
- ESM 蛋白质语言模型嵌入向量
- 最适生长温度 OGT（°C）

输出：
- 归一化热性能曲线（生长速率 vs 温度，峰值 = 1）
- UTPC 参数（Pmax、Topt、E）

---

## 目录结构

```
.
├── code/
│   ├── install_packages.py     # 一键安装所有依赖
│   ├── group4_train.py         # 主训练脚本（UTPC + 约束残差）
│   ├── group4_pinn_train.py    # PINN 训练脚本（ODE 约束）
│   ├── group4_predict.py       # 推理脚本（含使用教程）
│   └── 4groups.py              # 四组分类实验脚本
├── data/
│   └── ogt_simulator_pm5_by_curve_seed20260209.csv   # OGT 模拟数据（小文件）
│   # 注意：主训练数据集（~73 MB）体积过大，不含在仓库中，见下方"数据说明"
├── results/
│   ├── group4_pinn_checkpoint.pt   # 已训练模型权重
│   └── group4_pinn_scaler.pkl      # ESM 嵌入标准化器
├── Train/
│   ├── group4_pinn_training_log.csv       # 训练过程损失记录
│   ├── group4_pinn_training_loss.png      # 损失曲线图
│   └── group4_pinn_training_summary.json  # 训练汇总信息
├── requirements.txt
└── LICENSE
```

---

## 快速开始

### 1. 安装依赖

**方式 A — 一键脚本（推荐，自动安装 CUDA 版 PyTorch）：**
```bash
python code/install_packages.py
```

**方式 B — 手动安装：**
```bash
# CPU 版 PyTorch
pip install torch torchvision torchaudio

# 或 GPU 版（NVIDIA CUDA 12.x）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 其他依赖
pip install numpy pandas matplotlib scikit-learn xgboost pillow torchdiffeq
```

### 2. 训练模型

> 需要先获取训练数据集，放到 `data/` 目录（见下方"数据说明"）

```bash
python code/group4_train.py
```

训练完成后，模型权重保存在 `results/group4_checkpoint.pt`，标准化器保存在 `results/group4_scaler.pkl`。

**PINN 版本训练（含 ODE 约束）：**
```bash
python code/group4_pinn_train.py
```

### 3. 使用已训练模型进行预测

`results/` 目录已包含预训练的模型权重，可直接运行推理脚本：

```bash
python code/group4_predict.py
```

推理脚本顶部有详细的使用教程，支持**单条曲线预测**和**批量 CSV 预测**两种模式。

---

## 数据说明

主训练数据集文件名为：
```
11800TPC_1_1_with_medium_group_3_with_OGT (3).csv
```
文件体积约 73 MB，未包含在本仓库中。如需使用，请联系项目作者或参考原始数据来源。

获取文件后，将其放入 `data/` 目录即可。

---

## 模型架构

- **UTPC（Universal Thermal Performance Curve）** 参数化曲线作为物理先验
- **Transformer 注意力编码器** 提取 ESM 嵌入特征
- **ODE 约束残差模块** 通过 PINN 方式引入温度动力学约束
- 训练策略：Warmup → 交替优化（UTPC 参数 / 残差网络）→ 联合精调

---

## 环境要求

| 组件 | 版本要求 |
|------|---------|
| Python | 3.8+ |
| PyTorch | 2.0+（推荐 CUDA 12.x） |
| numpy | 任意 |
| pandas | 任意 |
| scikit-learn | 任意 |
| xgboost | 任意 |
| torchdiffeq | 任意（可选，缺少时自动回退内置 RK4） |

---

## License

[MIT](LICENSE)
