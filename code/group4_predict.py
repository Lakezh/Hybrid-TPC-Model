#!/usr/bin/env python3
"""
================================================================================
Group IV 模型推理脚本  —  使用教程
================================================================================

【功能】
  使用已训练好的 Group IV 模型（UTPC + 约束残差），对新输入数据进行预测，
  输出归一化的热性能曲线（生长速率 vs 温度）以及 UTPC 拟合参数。

【前提条件】
  1. 已运行 group4_train.py 完成训练，生成以下文件：
       ../results/group4_checkpoint.pt   （模型权重 + 元数据）
       ../results/group4_scaler.pkl      （ESM 嵌入标准化器）
  2. 安装依赖：pip install numpy pandas matplotlib scikit-learn torch

【输入说明】
  模型需要两类信息来为一条曲线进行预测：
    (a) ESM 嵌入向量  —  来自蛋白质语言模型，维度必须与训练时一致
    (b) OGT_sim_C     —  该生物的最适生长温度（摄氏度），用于锚定 Topt
  此外还需要：
    (c) 预测温度列表  —  希望在哪些温度点输出生长速率（摄氏度）

【输出说明】
  每条曲线返回一个字典，包含：
    pred_shape  —  归一化预测曲线（峰值=1），numpy array，与输入温度一一对应
    Pmax        —  UTPC 最大速率参数（浮点数）
    ToptC       —  最适温度（°C），等于输入的 OGT_sim_C（hard-anchor）
    E           —  UTPC 温度宽度参数（浮点数）

【两种使用方式】
  ① 单条曲线预测（直接传 numpy 数组）：见 predict_single()
  ② 批量预测（从 CSV 文件读入）       ：见 predict_from_csv()

【快速上手示例】
  见本文件底部 `if __name__ == "__main__":` 部分

================================================================================
"""

import math, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR      = Path(__file__).parent
RESULTS_DIR     = SCRIPT_DIR.parent / "results"
CHECKPOINT_PATH = RESULTS_DIR / "group4_checkpoint.pt"
SCALER_PATH     = RESULTS_DIR / "group4_scaler.pkl"

# ============================================================
# 模型架构（与 group4_train.py 完全一致，加载权重需要）
# ============================================================
ATTN_DIM = 128; N_HEADS = 4; N_LAYERS = 1; DROPOUT = 0.1; Z_DIM = 64
P_MAX = 10.0; E_MIN, E_MAX = 3.0, 60.0; X_MIN = -60.0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class ESMTempEncoder_MLP(nn.Module):
    def __init__(self, emb_len, attn_dim=ATTN_DIM, n_heads=N_HEADS,
                 n_layers=N_LAYERS, out_dim=Z_DIM, p_drop=DROPOUT,
                 n_patches=64, mlp_hidden=128, tfeat_dim=2):
        super().__init__()
        self.emb_len   = emb_len
        self.n_patches = n_patches
        self.patch_dim = emb_len // n_patches
        self.patch_mlp = nn.Sequential(
            nn.Linear(self.patch_dim, mlp_hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(mlp_hidden, attn_dim)
        )
        self.temp_proj = nn.Sequential(
            nn.Linear(tfeat_dim, attn_dim), nn.ReLU(), nn.Dropout(p_drop)
        )
        self.pos = PositionalEncoding(attn_dim, max_len=n_patches + 1)
        layer = nn.TransformerEncoderLayer(
            d_model=attn_dim, nhead=n_heads, dim_feedforward=attn_dim * 2,
            dropout=p_drop, batch_first=False, norm_first=True, activation='gelu'
        )
        self.tx  = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out = nn.Linear(attn_dim, out_dim)

    def forward(self, emb_vec, tfeat):
        B, L = emb_vec.shape
        x = self.patch_mlp(emb_vec.view(B, self.n_patches, self.patch_dim))
        temp_tok = self.temp_proj(tfeat).unsqueeze(1)
        x = torch.cat([temp_tok, x], dim=1).transpose(0, 1)
        return self.out(self.tx(self.pos(x))[0])


class ParamHead(nn.Module):
    def __init__(self, in_dim=Z_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(hidden, 2)
        )

    def forward(self, z):
        return self.net(z)


class ResidualMLP(nn.Module):
    def __init__(self, z_dim=Z_DIM, hidden=128, out_scale=1e-3, y_clip=50.0):
        super().__init__()
        self.y_clip = y_clip
        self.net = nn.Sequential(
            nn.Linear(2 + z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 1)
        )
        self.out_scale = out_scale

    def forward(self, t_norm, y, z):
        y_clip = torch.clamp(y, -self.y_clip, self.y_clip)
        x = torch.cat([t_norm.view(-1, 1), y_clip.view(-1, 1), z], dim=1)
        return self.net(x).squeeze(1) * self.out_scale


class UTPC_ODEFunc_Constrained(nn.Module):
    def __init__(self, residual_net, hard_gate=True, detach_z=True):
        super().__init__()
        self.residual  = residual_net
        self.hard_gate = hard_gate
        self.detach_z  = detach_z
        self.params = self.z = self.t_mean = self.t_std = None

    def set_context(self, Pmax, ToptC, E, z, t_mean, t_std):
        self.params = (Pmax, ToptC, E)
        self.z, self.t_mean, self.t_std = z, t_mean, t_std

    def forward(self, tK, y):
        t_in = tK.view(1) if tK.dim() == 0 else tK
        y_in = y.view(1)  if y.dim()  == 0 else y
        Tc   = t_in - 273.15
        Pmax, ToptC, E = self.params
        x_val = (Tc - ToptC) / (E + 1e-8)
        x_eff = torch.clamp(x_val, min=X_MIN, max=1.0)
        dbase = -(Pmax / (E + 1e-8)) * torch.exp(x_eff) * x_eff
        t_norm   = (t_in - self.t_mean) / (self.t_std + 1e-8)
        z_use    = self.z.detach() if self.detach_z else self.z
        dres_raw = self.residual(t_norm, y_in, z_use.expand(t_in.size(0), -1))
        dres     = torch.where(Tc > ToptC, -F.softplus(dres_raw), dres_raw) \
                   if self.hard_gate else dres_raw
        dy = dbase + dres
        dy = torch.where((x_val > 1.0) & (y_in <= 0.0), torch.zeros_like(dy), dy)
        return dy


class UDEModel_Constrained(nn.Module):
    def __init__(self, encoder, head, residual, hard_gate=True, detach_z=True):
        super().__init__()
        self.encoder = encoder
        self.head    = head
        self.odefunc = UTPC_ODEFunc_Constrained(residual, hard_gate, detach_z)

    def forward_curve(self, emb_std, Tk_vec, ogtK, t_mean_k, t_std_k, device):
        if not torch.is_tensor(emb_std): emb_std = torch.tensor(emb_std, dtype=torch.float32, device=device)
        if not torch.is_tensor(Tk_vec):  Tk_vec  = torch.tensor(Tk_vec,  dtype=torch.float32, device=device)
        if not torch.is_tensor(ogtK):    ogtK    = torch.tensor(float(ogtK), dtype=torch.float32, device=device)

        tfeat = torch.zeros((1, 2), dtype=torch.float32, device=device)
        z     = self.encoder(emb_std.view(1, -1), tfeat)
        raw   = self.head(z).view(-1)

        Pmax  = P_MAX * torch.sigmoid(raw[0]) + 1e-6
        E     = E_MIN + (E_MAX - E_MIN) * torch.sigmoid(raw[1])
        ToptC = ogtK - 273.15
        self.odefunc.set_context(Pmax, ToptC, E, z, t_mean_k, t_std_k)

        Tc0 = Tk_vec[:1] - 273.15
        x0  = torch.clamp((Tc0 - ToptC) / (E + 1e-8), min=X_MIN, max=1.0)
        y0  = torch.clamp(Pmax * torch.exp(x0) * (1.0 - x0), min=0.0).view(-1)

        try:
            from torchdiffeq import odeint as td_odeint
            traj = td_odeint(self.odefunc, y0, Tk_vec, rtol=1e-5, atol=1e-6, method="dopri5")
        except Exception:
            y = y0; ys = [y.clone()]
            for i in range(1, len(Tk_vec)):
                h  = (Tk_vec[i] - Tk_vec[i-1]).to(y.dtype)
                ti = Tk_vec[i-1]
                k1 = self.odefunc(ti,         y)
                k2 = self.odefunc(ti + 0.5*h, y + 0.5*h*k1)
                k3 = self.odefunc(ti + 0.5*h, y + 0.5*h*k2)
                k4 = self.odefunc(ti + h,     y + h*k3)
                y  = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
                ys.append(y.clone())
            traj = torch.stack(ys, dim=0)

        y_pred = torch.clamp(traj.squeeze(-1), min=0.0)
        return y_pred, (Pmax, ToptC, E)

# ============================================================
# 加载模型
# ============================================================

def load_model(checkpoint_path=CHECKPOINT_PATH, scaler_path=SCALER_PATH):
    """
    加载已训练好的 Group IV 模型。

    参数
    ----
    checkpoint_path : str 或 Path，默认 ../results/group4_checkpoint.pt
    scaler_path     : str 或 Path，默认 ../results/group4_scaler.pkl

    返回
    ----
    model, scaler, meta, device

    示例
    ----
    >>> model, scaler, meta, device = load_model()
    """
    checkpoint_path = Path(checkpoint_path)
    scaler_path     = Path(scaler_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"找不到模型文件：{checkpoint_path}\n请先运行 group4_train.py 完成训练。")
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"找不到 scaler 文件：{scaler_path}\n请先运行 group4_train.py 完成训练。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    hp     = ckpt.get("hyperparams", {})

    emb_len   = ckpt["emb_len"]
    n_patches = ckpt["n_patches"]
    z_dim     = hp.get("Z_DIM", Z_DIM)
    y_clip    = hp.get("Y_CLIP", 50.0)
    hard_gate = hp.get("HARD_NO_INCREASE_AFTER_OGT", True)
    attn_dim  = hp.get("ATTN_DIM", ATTN_DIM)
    n_heads   = hp.get("N_HEADS",  N_HEADS)
    n_layers  = hp.get("N_LAYERS", N_LAYERS)
    dropout   = hp.get("DROPOUT",  DROPOUT)

    encoder  = ESMTempEncoder_MLP(emb_len=emb_len, attn_dim=attn_dim, n_heads=n_heads,
                                  n_layers=n_layers, out_dim=z_dim, p_drop=dropout,
                                  n_patches=n_patches)
    head     = ParamHead(in_dim=z_dim)
    residual = ResidualMLP(z_dim=z_dim, y_clip=y_clip)

    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    residual.load_state_dict(ckpt["residual"])

    model = UDEModel_Constrained(encoder, head, residual, hard_gate=hard_gate)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    meta = {
        "t_mean_k": torch.tensor(ckpt["t_mean_k"], dtype=torch.float32, device=device),
        "t_std_k":  torch.tensor(ckpt["t_std_k"],  dtype=torch.float32, device=device),
        "esm_cols": ckpt["esm_cols"],
        "emb_len":  emb_len,
    }

    print(f"模型加载完成（device={device}，ESM dim={emb_len}）")
    return model, scaler, meta, device

# ============================================================
# 单条曲线预测
# ============================================================

def predict_single(model, scaler, meta, device,
                   esm_embedding, ogt_sim_c, temperatures):
    """
    对单条曲线（单个蛋白质）进行预测。

    参数
    ----
    esm_embedding : numpy array，形状 (emb_len,)，蛋白质的 ESM 嵌入向量
    ogt_sim_c     : float，该生物的最适生长温度（摄氏度），例如大肠杆菌 37.0
    temperatures  : numpy array 或 list，预测温度点（摄氏度，升序，至少 2 个）

    返回
    ----
    dict：pred_shape, Pmax, ToptC, E, temperatures

    示例
    ----
    >>> esm_emb = np.random.randn(1280).astype(np.float32)  # 替换为真实 ESM 嵌入
    >>> result  = predict_single(model, scaler, meta, device, esm_emb, 37.0,
    ...                          np.arange(10, 60, 5, dtype=np.float32))
    >>> print(result["pred_shape"])
    """
    esm_embedding = np.asarray(esm_embedding, dtype=np.float32).ravel()
    temperatures  = np.asarray(temperatures,  dtype=np.float32)

    if esm_embedding.shape[0] != meta["emb_len"]:
        raise ValueError(
            f"ESM 嵌入维度不匹配：输入 {esm_embedding.shape[0]}，"
            f"训练时为 {meta['emb_len']}")
    if len(temperatures) < 2:
        raise ValueError("temperatures 至少需要 2 个温度点")

    emb_std  = scaler.transform(esm_embedding.reshape(1, -1))[0].astype(np.float32)
    Tk_sorted = np.sort(temperatures + 273.15).astype(np.float32)
    ogtK      = float(ogt_sim_c) + 273.15

    with torch.no_grad():
        y_raw, (Pmax, ToptC, E) = model.forward_curve(
            emb_std, Tk_sorted, ogtK, meta["t_mean_k"], meta["t_std_k"], device)

    y_np = y_raw.cpu().numpy().astype(np.float32)
    peak = float(np.max(y_np))
    y_np = y_np / (peak if abs(peak) > 1e-8 else 1.0)

    return {
        "pred_shape":   y_np,
        "Pmax":         float(Pmax.item()),
        "ToptC":        float(ToptC.item()),
        "E":            float(E.item()),
        "temperatures": np.sort(temperatures),
    }

# ============================================================
# 批量预测（从 CSV 读取）
# ============================================================

def predict_from_csv(model, scaler, meta, device,
                     input_csv, output_csv=None,
                     col_id="TPC_id", col_temp="temperature",
                     col_ogt_sim="OGT_sim_C", col_species="binomial_name"):
    """
    从 CSV 文件批量预测。

    CSV 格式要求
    ------------
    必须包含：col_id, col_temp（摄氏度）, col_ogt_sim（摄氏度）, ESM 嵌入列
    可选列：col_species（物种名）

    参数
    ----
    input_csv  : 输入 CSV 路径
    output_csv : 输出路径（None 则不保存，只返回列表）

    返回
    ----
    list of dict，每条曲线含 TPC_id, T_C, pred_shape, Pmax, ToptC, E

    示例
    ----
    >>> results = predict_from_csv(model, scaler, meta, device,
    ...     input_csv="../data/my_curves.csv",
    ...     output_csv="../data/predictions.csv")
    """
    import json as _json
    df = pd.read_csv(input_csv, low_memory=False)
    esm_cols = meta["esm_cols"]

    missing = [c for c in [col_id, col_temp, col_ogt_sim] + esm_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少以下列：{missing}")

    df[col_temp]    = pd.to_numeric(df[col_temp],    errors="coerce")
    df[col_ogt_sim] = pd.to_numeric(df[col_ogt_sim], errors="coerce")
    df[col_id]      = df[col_id].astype(str)
    df              = df.dropna(subset=[col_temp, col_ogt_sim] + esm_cols).reset_index(drop=True)

    Xemb_std = scaler.transform(df[esm_cols].values).astype(np.float32)

    results = []
    for tid, sub in df.groupby(col_id):
        sub  = sub.sort_values(col_temp).reset_index(drop=True)
        Tc   = sub[col_temp].values.astype(np.float32)
        ogtC = float(sub[col_ogt_sim].iloc[0])
        emb_std_row = Xemb_std[df[df[col_id] == tid].index[0]]
        Tk_array    = np.sort(Tc + 273.15).astype(np.float32)
        species     = str(sub.iloc[0][col_species]) if col_species in sub.columns else "unknown"

        with torch.no_grad():
            y_raw, (Pmax, ToptC, E) = model.forward_curve(
                emb_std_row, Tk_array, ogtC + 273.15,
                meta["t_mean_k"], meta["t_std_k"], device)

        y_np = y_raw.cpu().numpy().astype(np.float32)
        peak = float(np.max(y_np))
        y_np = y_np / (peak if abs(peak) > 1e-8 else 1.0)

        results.append({
            col_id:       tid,
            "species":    species,
            "T_C":        np.sort(Tc).tolist(),
            "pred_shape": y_np.tolist(),
            "Pmax":       float(Pmax.item()),
            "ToptC":      float(ToptC.item()),
            "E":          float(E.item()),
        })

    print(f"完成预测：共 {len(results)} 条曲线")

    if output_csv is not None:
        out_rows = [{
            col_id:         r[col_id],
            "species":      r["species"],
            "T_C":          _json.dumps(r["T_C"]),
            "pred_shape":   _json.dumps(r["pred_shape"]),
            "Pmax":         r["Pmax"],
            "ToptC":        r["ToptC"],
            "E":            r["E"],
        } for r in results]
        pd.DataFrame(out_rows).to_csv(output_csv, index=False)
        print(f"预测结果已保存到：{output_csv}")

    return results

# ============================================================
# 可视化
# ============================================================

def plot_prediction(result, title=None, save_path=None):
    """绘制单条曲线的预测结果图。"""
    T   = result["temperatures"]
    y   = result["pred_shape"]
    opt = result["ToptC"]

    plt.figure(figsize=(7, 4))
    plt.plot(T, y, "o--", color="royalblue", linewidth=2, label="预测曲线 (归一化)")
    plt.axvline(opt, color="tomato", linestyle=":", linewidth=1.5,
                label=f"Topt = {opt:.1f} C")
    plt.xlabel("温度 (°C)"); plt.ylabel("归一化生长速率")
    plt.title(title or "Group IV 预测结果")
    plt.legend(fontsize=9); plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图已保存：{save_path}")
    else:
        plt.show()
    plt.close()

# ============================================================
# 快速上手示例
# ============================================================

if __name__ == "__main__":

    # 步骤 1：加载模型（每次程序启动只需加载一次）
    model, scaler, meta, device = load_model()

    # 场景 A：单条曲线预测
    print("\n=== 场景 A：单条曲线预测 ===")
    emb_len = meta["emb_len"]
    print(f"提示：训练使用的 ESM 嵌入维度为 {emb_len}")

    # 【替换这里】换成你的真实 ESM 嵌入向量（shape: (emb_len,)）
    fake_esm_embedding = np.random.randn(emb_len).astype(np.float32)

    # 【替换这里】设置该生物的 OGT（摄氏度）
    ogt_sim_c = 37.0

    # 【替换这里】设置预测温度范围
    temperatures = np.arange(10, 60, 5, dtype=np.float32)

    result_A = predict_single(
        model, scaler, meta, device,
        esm_embedding=fake_esm_embedding,
        ogt_sim_c=ogt_sim_c,
        temperatures=temperatures
    )
    print(f"预测温度点：{result_A['temperatures']}")
    print(f"预测归一化曲线：{np.round(result_A['pred_shape'], 3)}")
    print(f"UTPC 参数：Pmax={result_A['Pmax']:.4f}, ToptC={result_A['ToptC']:.2f}C, E={result_A['E']:.2f}")

    # 场景 B：CSV 批量预测（取消注释使用）
    # print("\n=== 场景 B：CSV 批量预测 ===")
    # results_B = predict_from_csv(
    #     model, scaler, meta, device,
    #     input_csv  = "../data/my_new_curves.csv",
    #     output_csv = "../data/my_predictions.csv",
    # )
    # print(f"共预测 {len(results_B)} 条曲线")

    print("\n推理脚本运行完毕。")