#!/usr/bin/env python3
"""
Group IV — UTPC + 约束残差  全量训练脚本
使用全部可用数据训练，不做 cross-validation。
训练完成后将模型权重和预处理器保存到 ../results/ 目录。
"""

import os, sys, math, random, warnings, json, hashlib, time, pickle
# Force UTF-8 output on Windows consoles that default to cp932/gbk
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV        = DATA_DIR / "11800TPC_1_1_with_medium_group_3_with_OGT (3).csv"
SIM_FILE        = DATA_DIR / "ogt_simulator_pm5_by_curve_seed20260209.csv"
CHECKPOINT_PATH = RESULTS_DIR / "group4_checkpoint.pt"
SCALER_PATH     = RESULTS_DIR / "group4_scaler.pkl"

# ============================================================
# 随机种子 & 设备
# ============================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# 超参数
# ============================================================
ATTN_DIM = 128
N_HEADS  = 4
N_LAYERS = 1
DROPOUT  = 0.1
Z_DIM    = 64

P_MAX        = 10.0
E_MIN, E_MAX = 3.0, 60.0
X_MIN        = -60.0

WARMUP_EPOCHS    = 25
ALT_CYCLES       = 4
ALT_EPOCHS_THETA = 8
ALT_EPOCHS_RESID = 8
JOINT_EPOCHS     = 20
LR_THETA         = 1e-3
LR_RESID         = 2e-3
WEIGHT_DECAY     = 1e-4
CLIP_NORM        = 1.0

RES_LAMBDA                 = 1e-3
DETACH_Z                   = True
Y_CLIP                     = 50.0
HARD_NO_INCREASE_AFTER_OGT = True
LAMBDA_MONO                = 0.2
LAMBDA_TAIL                = 0.05
TAIL_TARGET                = 0.0

COL_ID      = "TPC_id"
COL_SPECIES = "binomial_name"
COL_TEMP    = "temperature"
COL_Y       = "mu"
COL_OGT     = "OGT"
COL_OGT_SIM = "OGT_sim_C"
COL_KINGDOM = "kingdom"
COL_ESM_GRP = "esm_group"
KINGDOM_KEEP = {"Bacteria", "Archaea", "Eubacteria"}
SIM_SEED     = 20260209

# ============================================================
# 工具函数
# ============================================================

def compute_curve_shape_max_anchor(y):
    y = np.asarray(y, np.float32)
    m = float(np.max(y))
    denom = m if abs(m) > 1e-8 else 1.0
    return (y / denom).astype(np.float32)


def pearsonr_np(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    y0 = y - y.mean(); y1 = yhat - yhat.mean()
    denom = np.linalg.norm(y0) * np.linalg.norm(y1) + 1e-12
    return float((y0 @ y1) / denom) if denom > 1e-12 else 0.0


def pearson_r2_np(y, yhat):
    return float(pearsonr_np(y, yhat) ** 2)


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def stable_hash_float_row(arr_1d: np.ndarray, ndigits: int = 6) -> str:
    x = np.round(np.asarray(arr_1d, dtype=np.float32), ndigits)
    return hashlib.sha1(x.tobytes()).hexdigest()


def deterministic_noise_pm5(tpc_id: str, seed_: int = 20260209) -> float:
    key = f"{seed_}|{tpc_id}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    u = int(h[:16], 16) / float(16**16 - 1)
    return -5.0 + 10.0 * u


def odeint_rk4(func, y0, t):
    y = y0.view(-1)
    ys = [y.clone()]
    for i in range(1, len(t)):
        h  = (t[i] - t[i - 1]).to(y.dtype)
        ti = t[i - 1]
        k1 = func(ti,            y)
        k2 = func(ti + 0.5 * h, y + 0.5 * h * k1)
        k3 = func(ti + 0.5 * h, y + 0.5 * h * k2)
        k4 = func(ti + h,       y + h * k3)
        y  = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ys.append(y.clone())
    return torch.stack(ys, dim=0)


def utpc_rate_torch(Tc, Pmax, ToptC, E):
    x      = (Tc - ToptC) / (E + 1e-8)
    x_safe = torch.clamp(x, min=X_MIN, max=1.0)
    y_pos  = torch.exp(x_safe) * (1.0 - x_safe)
    y      = torch.where(x <= 1.0, y_pos, torch.zeros_like(y_pos))
    return Pmax * torch.clamp(y, min=0.0)


def utpc_drate_dT_torch(Tc, Pmax, ToptC, E):
    x     = (Tc - ToptC) / (E + 1e-8)
    x_eff = torch.clamp(x, min=X_MIN, max=1.0)
    return -(Pmax / (E + 1e-8)) * torch.exp(x_eff) * x_eff


def choose_n_patches(emb_len: int) -> int:
    for p in [128, 64, 32, 16, 8, 4, 2, 1]:
        if emb_len % p == 0:
            return p
    return 1


def map_params(raw_vec, ogtK):
    Pmax  = P_MAX * torch.sigmoid(raw_vec[0]) + 1e-6
    E     = E_MIN + (E_MAX - E_MIN) * torch.sigmoid(raw_vec[1])
    ToptC = ogtK - 273.15
    return Pmax, ToptC, E


def build_curves(frame, Xemb_std, esm_cols,
                 col_id, col_temp, col_y, col_ogt_sim, col_species):
    curves = {}
    for tid, sub in frame.groupby(col_id):
        sub  = sub.sort_values(col_temp)
        Tk   = (sub[col_temp].values.astype(np.float32) + 273.15)
        y    = sub[col_y].values.astype(np.float32)
        ogtC = min(max(float(sub[col_ogt_sim].iloc[0]),
                       float(sub[col_temp].min())),
                   float(sub[col_temp].max()))
        curves[tid] = dict(
            Tk=Tk, y=y,
            y_shape=compute_curve_shape_max_anchor(y),
            emb=Xemb_std[int(sub.index.values[0])],
            ogtK=np.float32(ogtC + 273.15),
            species=sub.iloc[0][col_species],
            ogt_sim_c=float(ogtC)
        )
    return curves

# ============================================================
# 神经网络模块
# ============================================================

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
        assert emb_len % n_patches == 0, \
            f"emb_len={emb_len} 必须能被 n_patches={n_patches} 整除"
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
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.out_scale = out_scale

    def forward(self, t_norm, y, z):
        y_clip = torch.clamp(y, -self.y_clip, self.y_clip)
        x = torch.cat([t_norm.view(-1, 1), y_clip.view(-1, 1), z], dim=1)
        return self.net(x).squeeze(1) * self.out_scale


class UTPC_ODEFunc_Constrained(nn.Module):
    def __init__(self, residual_net, detach_z=True):
        super().__init__()
        self.residual = residual_net
        self.detach_z = detach_z
        self.params = self.z = self.t_mean = self.t_std = None

    def set_context(self, Pmax, ToptC, E, z, t_mean, t_std):
        self.params = (Pmax, ToptC, E)
        self.z, self.t_mean, self.t_std = z, t_mean, t_std

    def forward(self, tK, y):
        t_in = tK.view(1) if tK.dim() == 0 else tK
        y_in = y.view(1)  if y.dim()  == 0 else y
        Tc   = t_in - 273.15
        Pmax, ToptC, E = self.params
        dbase    = utpc_drate_dT_torch(Tc, Pmax, ToptC, E)
        t_norm   = (t_in - self.t_mean) / (self.t_std + 1e-8)
        z_use    = self.z.detach() if self.detach_z else self.z
        dres_raw = self.residual(t_norm, y_in, z_use.expand(t_in.size(0), -1))
        if HARD_NO_INCREASE_AFTER_OGT:
            dres = torch.where(Tc > ToptC, -F.softplus(dres_raw), dres_raw)
        else:
            dres = dres_raw
        dy = dbase + dres
        x  = (Tc - ToptC) / (E + 1e-8)
        dy = torch.where((x > 1.0) & (y_in <= 0.0), torch.zeros_like(dy), dy)
        return dy


class UDEModel_Constrained(nn.Module):
    def __init__(self, encoder, head, residual):
        super().__init__()
        self.encoder = encoder
        self.head    = head
        self.odefunc = UTPC_ODEFunc_Constrained(residual, detach_z=DETACH_Z)

    def forward_curve(self, emb_std, Tk_vec, ogtK, t_mean_k, t_std_k):
        if not torch.is_tensor(emb_std): emb_std = torch.tensor(emb_std, dtype=torch.float32, device=device)
        if not torch.is_tensor(Tk_vec):  Tk_vec  = torch.tensor(Tk_vec,  dtype=torch.float32, device=device)
        if not torch.is_tensor(ogtK):    ogtK    = torch.tensor(ogtK,    dtype=torch.float32, device=device)

        tfeat = torch.zeros((1, 2), dtype=torch.float32, device=device)
        z     = self.encoder(emb_std.view(1, -1), tfeat)
        raw   = self.head(z).view(-1)
        Pmax, ToptC, E = map_params(raw, ogtK)
        self.odefunc.set_context(Pmax, ToptC, E, z, t_mean_k, t_std_k)

        y0 = utpc_rate_torch(Tk_vec[:1] - 273.15, Pmax, ToptC, E).view(-1)
        try:
            from torchdiffeq import odeint as td_odeint
            traj = td_odeint(self.odefunc, y0, Tk_vec, rtol=1e-5, atol=1e-6, method="dopri5")
        except Exception:
            traj = odeint_rk4(self.odefunc, y0, Tk_vec)
        y_pred = torch.clamp(traj.squeeze(-1), min=0.0)

        t_norm    = (Tk_vec - t_mean_k) / (t_std_k + 1e-8)
        z_use     = z.detach() if self.odefunc.detach_z else z
        resid_seq = self.odefunc.residual(t_norm, y_pred, z_use.expand(Tk_vec.shape[0], -1))
        return y_pred, resid_seq, (Pmax, ToptC, E)

# ============================================================
# 数据加载
# ============================================================

def load_data():
    df_raw = pd.read_csv(DATA_CSV, low_memory=False)
    for c in [COL_ID, COL_SPECIES, COL_TEMP, COL_KINGDOM, COL_OGT]:
        assert c in df_raw.columns, f"缺少必要列: {c}"
    assert COL_Y in df_raw.columns

    df_raw[COL_KINGDOM] = df_raw[COL_KINGDOM].astype(str)
    df_raw = df_raw[df_raw[COL_KINGDOM].isin(KINGDOM_KEEP)].copy()

    df_raw[COL_TEMP] = pd.to_numeric(df_raw[COL_TEMP], errors="coerce")
    df_raw[COL_Y]    = pd.to_numeric(df_raw[COL_Y],    errors="coerce").fillna(0.0)
    df_raw[COL_OGT]  = pd.to_numeric(df_raw[COL_OGT],  errors="coerce")

    ogt_med = df_raw[COL_OGT].median()
    if pd.notna(ogt_med) and ogt_med > 150:
        print("[INFO] OGT 疑为 Kelvin，自动转换 OGT -= 273.15")
        df_raw[COL_OGT] -= 273.15

    tmp = (
        df_raw.groupby([COL_ID, COL_TEMP], as_index=False)[COL_Y].mean()
              .sort_values([COL_ID, COL_Y], ascending=[True, False])
              .groupby(COL_ID, as_index=False).head(1)
              .rename(columns={COL_TEMP: "OGT_fill"})
    )
    df_raw = df_raw.merge(tmp[[COL_ID, "OGT_fill"]], on=COL_ID, how="left")
    df_raw[COL_OGT] = df_raw[COL_OGT].fillna(df_raw["OGT_fill"])
    df_raw.drop(columns=["OGT_fill"], inplace=True)

    esm_cols = [
        c for c in df_raw.columns
        if c.lower().startswith("esm") and pd.api.types.is_numeric_dtype(df_raw[c])
    ]
    assert len(esm_cols) > 0, "未找到以 'esm' 开头的数值型列"
    print(f"ESM dims = {len(esm_cols)}")

    before = len(df_raw)
    df_raw = df_raw.dropna(subset=esm_cols, how="any").reset_index(drop=True)
    print(f"删除 ESM 缺失行: {before} → {len(df_raw)}")

    df_y   = df_raw.groupby([COL_ID, COL_SPECIES, COL_TEMP], as_index=False)[COL_Y].mean()
    ogt_df = df_raw.groupby([COL_ID, COL_SPECIES], as_index=False)[COL_OGT].mean()
    emb_df = (
        df_raw[[COL_ID, COL_SPECIES] + esm_cols]
        .drop_duplicates([COL_ID, COL_SPECIES])
        .groupby([COL_ID, COL_SPECIES], as_index=False).mean()
    )
    df = (
        df_y.merge(emb_df, on=[COL_ID, COL_SPECIES], how="left")
            .merge(ogt_df, on=[COL_ID, COL_SPECIES], how="left")
    )
    df[COL_ID] = df[COL_ID].astype(str)
    assert df[esm_cols].isna().any(axis=1).sum() == 0
    assert df[COL_OGT].isna().sum() == 0

    curve_ogt_base = (
        df[[COL_ID, COL_OGT]].drop_duplicates(COL_ID)
        .groupby(COL_ID, as_index=False)[COL_OGT].mean()
    )
    curve_ogt_base[COL_ID] = curve_ogt_base[COL_ID].astype(str)

    if SIM_FILE.exists() and SIM_FILE.stat().st_size > 0:
        sim_df = pd.read_csv(SIM_FILE)
        sim_df[COL_ID] = sim_df[COL_ID].astype(str)
        miss_ids = sorted(set(curve_ogt_base[COL_ID]) - set(sim_df[COL_ID]))
        if miss_ids:
            base_map = dict(zip(curve_ogt_base[COL_ID], curve_ogt_base[COL_OGT]))
            new_rows = [{
                COL_ID: tid, "OGT_orig_C": base_map[tid],
                "delta_C": deterministic_noise_pm5(tid, SIM_SEED),
                COL_OGT_SIM: base_map[tid] + deterministic_noise_pm5(tid, SIM_SEED)
            } for tid in miss_ids]
            sim_df = pd.concat([sim_df, pd.DataFrame(new_rows)], ignore_index=True)
            sim_df.drop_duplicates(subset=[COL_ID], keep="first", inplace=True)
            sim_df.to_csv(SIM_FILE, index=False)
    else:
        rows = []
        for _, r in curve_ogt_base.sort_values(COL_ID).iterrows():
            tid = str(r[COL_ID]); ogt0 = float(r[COL_OGT])
            delta = deterministic_noise_pm5(tid, SIM_SEED)
            rows.append({COL_ID: tid, "OGT_orig_C": ogt0, "delta_C": delta,
                         COL_OGT_SIM: ogt0 + delta})
        sim_df = pd.DataFrame(rows)
        sim_df.to_csv(SIM_FILE, index=False)
        print(f"[INFO] 创建 OGT 模拟器文件: {SIM_FILE}")

    df = df.merge(sim_df[[COL_ID, COL_OGT_SIM]], on=COL_ID, how="left")
    assert df[COL_OGT_SIM].isna().sum() == 0

    print(f"数据就绪：{len(df)} 行 | {df[COL_ID].nunique()} 条曲线 | {df[COL_SPECIES].nunique()} 个物种")
    return df, esm_cols

# ============================================================
# 训练（全量数据）
# ============================================================

def train_all(df, esm_cols):
    print("\n=== Group IV — UTPC + 约束残差 | 全量训练 ===")
    t0 = time.time()

    emb_scaler = StandardScaler().fit(df[esm_cols].values)
    Xemb_std   = emb_scaler.transform(df[esm_cols].values).astype(np.float32)
    all_curves = build_curves(df, Xemb_std, esm_cols,
                              COL_ID, COL_TEMP, COL_Y, COL_OGT_SIM, COL_SPECIES)

    K_all    = (df[COL_TEMP].values.astype(np.float32) + 273.15)
    t_mean_k = torch.tensor(float(K_all.mean()), device=device)
    t_std_k  = torch.tensor(float(K_all.std() + 1e-8), device=device)

    EMB_LEN   = len(esm_cols)
    n_patches = choose_n_patches(EMB_LEN)
    encoder   = ESMTempEncoder_MLP(emb_len=EMB_LEN, n_patches=n_patches).to(device)
    head      = ParamHead().to(device)
    residual  = ResidualMLP(z_dim=Z_DIM, y_clip=Y_CLIP).to(device)
    model     = UDEModel_Constrained(encoder, head, residual).to(device)

    theta_params = list(encoder.parameters()) + list(head.parameters())
    resid_params = list(residual.parameters())
    opt_theta = torch.optim.Adam(theta_params, lr=LR_THETA, weight_decay=WEIGHT_DECAY)
    opt_resid = torch.optim.Adam(resid_params, lr=LR_RESID, weight_decay=WEIGHT_DECAY)
    loss_fn   = nn.SmoothL1Loss()
    log_rows  = []
    ep_counter = {"v": 0}

    def run_epoch(train_theta, train_resid, tag):
        ep_counter["v"] += 1
        model.train()
        set_requires_grad(encoder,  train_theta)
        set_requires_grad(head,     train_theta)
        set_requires_grad(residual, train_resid)
        total_data = total_reg = total_mono = total_tail = total = ncur = 0

        for tid in random.sample(list(all_curves), len(all_curves)):
            C    = all_curves[tid]
            Tk   = torch.tensor(C["Tk"],      dtype=torch.float32, device=device)
            ogtK = torch.tensor(C["ogtK"],    dtype=torch.float32, device=device)
            y_ts = torch.tensor(C["y_shape"], dtype=torch.float32, device=device)
            emb  = torch.tensor(C["emb"],     dtype=torch.float32, device=device)

            opt_theta.zero_grad(set_to_none=True)
            opt_resid.zero_grad(set_to_none=True)

            y_raw, resid_seq, _ = model.forward_curve(emb, Tk, ogtK, t_mean_k, t_std_k)
            y_pred    = y_raw / (torch.max(y_raw).abs() + 1e-8)
            loss_data = loss_fn(y_pred, y_ts)
            loss_reg  = torch.mean(resid_seq ** 2)

            post_pair = (Tk[1:] >= ogtK) & (Tk[:-1] >= ogtK)
            loss_mono = torch.mean(torch.relu(y_pred[1:] - y_pred[:-1])[post_pair]) \
                        if torch.any(post_pair) else torch.tensor(0.0, device=device)
            loss_tail  = (y_pred[-1] - TAIL_TARGET) ** 2
            loss_total = loss_data + LAMBDA_MONO * loss_mono + LAMBDA_TAIL * loss_tail
            if train_resid:
                loss_total = loss_total + RES_LAMBDA * loss_reg

            loss_total.backward()
            if train_theta:
                torch.nn.utils.clip_grad_norm_(theta_params, CLIP_NORM)
                opt_theta.step()
            if train_resid:
                torch.nn.utils.clip_grad_norm_(resid_params, CLIP_NORM)
                opt_resid.step()

            total_data += float(loss_data.item())
            total_reg  += float(loss_reg.item())
            total_mono += float(loss_mono.item())
            total_tail += float(loss_tail.item())
            total      += float(loss_total.item())
            ncur       += 1

        avg_d, avg_r, avg_m, avg_tl, avg_t = [
            v / max(1, ncur)
            for v in [total_data, total_reg, total_mono, total_tail, total]
        ]
        print(f"  {tag} | total {avg_t:.5f}  (data {avg_d:.5f} | reg {avg_r:.5f} | "
              f"mono {avg_m:.5f} | tail {avg_tl:.5f})")
        log_rows.append({
            "ep": ep_counter["v"], "tag": tag,
            "avg_loss_data": avg_d, "avg_loss_reg": avg_r,
            "avg_loss_mono": avg_m, "avg_loss_tail": avg_tl, "avg_loss_total": avg_t
        })

    for ep in range(1, WARMUP_EPOCHS + 1):
        run_epoch(True, False, f"[Warmup θ] Ep {ep:02d}")

    for cyc in range(1, ALT_CYCLES + 1):
        for ep in range(1, ALT_EPOCHS_THETA + 1):
            run_epoch(True,  False, f"[Alt{cyc} θ] Ep {ep:02d}")
        for ep in range(1, ALT_EPOCHS_RESID + 1):
            run_epoch(False, True,  f"[Alt{cyc} resid] Ep {ep:02d}")

    for g in opt_theta.param_groups: g['lr'] = LR_THETA * 0.3
    for g in opt_resid.param_groups: g['lr'] = LR_RESID * 0.3
    for ep in range(1, JOINT_EPOCHS + 1):
        run_epoch(True, True, f"[Joint] Ep {ep:02d}")

    elapsed = time.time() - t0
    print(f"\n训练完成，用时 {elapsed:.1f}s")

    log_path = RESULTS_DIR / "group4_training_log.csv"
    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"训练日志已保存: {log_path}")

    log_df = pd.DataFrame(log_rows)
    plt.figure(figsize=(10, 4))
    plt.plot(log_df["ep"], log_df["avg_loss_total"], label="total")
    plt.plot(log_df["ep"], log_df["avg_loss_data"],  label="data")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Group IV Training Loss")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "group4_training_loss.png", dpi=150)
    plt.close()
    print(f"损失曲线已保存: {RESULTS_DIR / 'group4_training_loss.png'}")

    return model, emb_scaler, t_mean_k, t_std_k, EMB_LEN, n_patches

# ============================================================
# 保存模型
# ============================================================

def save_model(model, emb_scaler, t_mean_k, t_std_k, EMB_LEN, n_patches, esm_cols):
    torch.save({
        "encoder":   model.encoder.state_dict(),
        "head":      model.head.state_dict(),
        "residual":  model.odefunc.residual.state_dict(),
        "emb_len":   EMB_LEN,
        "n_patches": n_patches,
        "t_mean_k":  float(t_mean_k.cpu()),
        "t_std_k":   float(t_std_k.cpu()),
        "esm_cols":  esm_cols,
        "hyperparams": {
            "ATTN_DIM": ATTN_DIM, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS,
            "DROPOUT": DROPOUT, "Z_DIM": Z_DIM,
            "Y_CLIP": Y_CLIP, "HARD_NO_INCREASE_AFTER_OGT": HARD_NO_INCREASE_AFTER_OGT,
        }
    }, CHECKPOINT_PATH)
    print(f"模型已保存: {CHECKPOINT_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(emb_scaler, f)
    print(f"Scaler 已保存: {SCALER_PATH}")

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    df, esm_cols = load_data()
    model, emb_scaler, t_mean_k, t_std_k, EMB_LEN, n_patches = train_all(df, esm_cols)
    save_model(model, emb_scaler, t_mean_k, t_std_k, EMB_LEN, n_patches, esm_cols)
    print("\n完成。模型文件保存在:", RESULTS_DIR)