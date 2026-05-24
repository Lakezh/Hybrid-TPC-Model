#!/usr/bin/env python3
"""
Group IV PINN — UTPC + Biophysical Constraints  Full Training Script

Based on group4_train.py. Adds a minimal set of PINN-style biophysical constraints
for TPC (Thermal Performance Curve) prediction:

  1. Non-negativity (hard constraint): softplus replaces clamp(min=0) in model output
  2. Arrhenius rising-phase constraint (L_arrhenius): penalise curvature of
     log(y_hat) w.r.t. 1/T_K for T < Topt  [lambda = 0.01]
  3. Peak-height constraint (L_peak_height): y_hat(Topt) should be near max(y_hat)
     [lambda = 0.05]
  4. Post-peak tail-decline constraint (L_tail_decline): no positive slopes after Topt
     [lambda = 0.10]

Total loss:
    L_total = L_data + 0.01*L_arr + 0.05*L_peak + 0.10*L_tail  (+RES_LAMBDA*L_reg)

Training records  → ../Train/
Model / scaler    → ../results/
"""

import math, random, warnings, hashlib, time, pickle
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

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"
TRAIN_DIR   = SCRIPT_DIR.parent / "Train"
for _d in [RESULTS_DIR, TRAIN_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DATA_CSV        = DATA_DIR / "11800TPC_1_1_with_medium_group_3_with_OGT (3).csv"
SIM_FILE        = DATA_DIR / "ogt_simulator_pm5_by_curve_seed20260209.csv"
CHECKPOINT_PATH = RESULTS_DIR / "group4_pinn_checkpoint.pt"
SCALER_PATH     = RESULTS_DIR / "group4_pinn_scaler.pkl"

# ──────────────────────────────────────────────────────────────
# Random seed & device
# ──────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
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

# ──────────────────────────────────────────────────────────────
# Model architecture hyperparameters (same as group4)
# ──────────────────────────────────────────────────────────────
ATTN_DIM     = 128
N_HEADS      = 4
N_LAYERS     = 1
DROPOUT      = 0.1
Z_DIM        = 64
P_MAX        = 10.0
E_MIN, E_MAX = 3.0, 60.0
X_MIN        = -60.0

# ──────────────────────────────────────────────────────────────
# Training schedule hyperparameters (same as group4)
# ──────────────────────────────────────────────────────────────
WARMUP_EPOCHS    = 25
ALT_CYCLES       = 4
ALT_EPOCHS_THETA = 8
ALT_EPOCHS_RESID = 8
JOINT_EPOCHS     = 20
LR_THETA         = 1e-3
LR_RESID         = 2e-3
WEIGHT_DECAY     = 1e-4
CLIP_NORM        = 1.0
RES_LAMBDA       = 1e-3
DETACH_Z         = True
Y_CLIP           = 50.0
HARD_NO_INCREASE_AFTER_OGT = True

# ──────────────────────────────────────────────────────────────
# PINN constraint weights
# ──────────────────────────────────────────────────────────────
LAMBDA_ARR  = 0.01   # weak  — real TPCs are noisy, Arrhenius only approximate
LAMBDA_PEAK = 0.05   # moderate — Topt/OGT carries uncertainty
LAMBDA_TAIL = 0.10   # strong — high-T tail upturn is biologically unrealistic

# ──────────────────────────────────────────────────────────────
# Column names
# ──────────────────────────────────────────────────────────────
COL_ID       = "TPC_id"
COL_SPECIES  = "binomial_name"
COL_TEMP     = "temperature"
COL_Y        = "mu"
COL_OGT      = "OGT"
COL_OGT_SIM  = "OGT_sim_C"
COL_KINGDOM  = "kingdom"
KINGDOM_KEEP = {"Bacteria", "Archaea", "Eubacteria"}
SIM_SEED     = 20260209


# ══════════════════════════════════════════════════════════════
# Utility helpers
# ══════════════════════════════════════════════════════════════

def compute_curve_shape_max_anchor(y):
    y = np.asarray(y, np.float32)
    m = float(np.max(y))
    denom = m if abs(m) > 1e-8 else 1.0
    return (y / denom).astype(np.float32)


def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def deterministic_noise_pm5(tpc_id: str, seed_: int = 20260209) -> float:
    key = f"{seed_}|{tpc_id}".encode("utf-8")
    h   = hashlib.sha256(key).hexdigest()
    u   = int(h[:16], 16) / float(16**16 - 1)
    return -5.0 + 10.0 * u


def odeint_rk4(func, y0, t):
    y = y0.view(-1); ys = [y.clone()]
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
            ogt_sim_c=float(ogtC),
        )
    return curves


# ══════════════════════════════════════════════════════════════
# PINN Biophysical Constraint Losses
# ══════════════════════════════════════════════════════════════

def compute_arrhenius_loss(y_hat: torch.Tensor,
                           Tk: torch.Tensor,
                           ogtK: torch.Tensor,
                           eps: float = 1e-6) -> torch.Tensor:
    """
    Penalise non-linearity of log(y_hat) w.r.t. 1/T_K for pre-peak points (T < Topt).

    Arrhenius kinetics predict log(rate) is linear in 1/T_K.
    We measure the fraction of variance in log(y_hat) NOT explained by a linear
    fit against 1/T_K, i.e. (1 - R²).  This is mathematically equivalent to
    penalising curvature, but is dimensionless and bounded in [0, 1], avoiding
    the extreme numerical scale of d²log(y)/d(1/T)² (which has units K² and
    can reach 10¹¹ for typical temperature grids).

    Returns 0 when log(y_hat) is perfectly linear in 1/T_K (pure Arrhenius),
    returns 1 when there is no linear correlation at all.
    """
    mask_pre = Tk < ogtK
    if int(mask_pre.sum().item()) < 3:
        return torch.tensor(0.0, device=y_hat.device)

    x = 1.0 / Tk[mask_pre]                # 1/T_K
    z = torch.log(y_hat[mask_pre] + eps)  # log(y_hat + eps)

    # OLS: fit z = a + b*x, then measure residual variance / total variance
    x_c  = x - x.mean()
    z_c  = z - z.mean()
    ss_xz = (x_c * z_c).sum()
    ss_xx = (x_c * x_c).sum().clamp(min=1e-12)
    ss_zz = (z_c * z_c).sum().clamp(min=1e-12)

    r2 = (ss_xz ** 2) / (ss_xx * ss_zz)   # coefficient of determination
    return (1.0 - r2.clamp(max=1.0)).clamp(min=0.0)


def compute_peak_height_loss(y_hat: torch.Tensor,
                              Tk: torch.Tensor,
                              ogtK: torch.Tensor) -> torch.Tensor:
    """
    Encourage y_hat evaluated at Topt to equal max(y_hat) over the whole curve.

    After max-normalisation y_hat has max ≈ 1.  This loss pushes the value
    at the nearest observed temperature point to Topt towards the global max,
    i.e. it encourages the peak to sit at Topt rather than elsewhere.

    Using  (y_hat(Topt) - max(y_hat))²  rather than  (y_hat(Topt) - 1)²  so
    that the constraint remains valid even when Topt is noisy.
    """
    dist     = torch.abs(Tk - ogtK)
    idx_opt  = torch.argmin(dist)
    y_at_opt = y_hat[idx_opt]
    y_max    = torch.max(y_hat)
    return (y_at_opt - y_max) ** 2


def compute_tail_decline_loss(y_hat: torch.Tensor,
                               Tk: torch.Tensor,
                               ogtK: torch.Tensor) -> torch.Tensor:
    """
    Penalise positive slopes in y_hat for intervals where the midpoint T > Topt.

    Finite-difference slopes are computed over consecutive sorted temperature
    points.  ReLU is applied so only upward slopes contribute to the loss.
    Returns zero if there are no post-Topt intervals.
    """
    if len(Tk) < 2:
        return torch.tensor(0.0, device=y_hat.device)

    dT    = (Tk[1:] - Tk[:-1]).clamp(min=1e-8)
    dy    = y_hat[1:] - y_hat[:-1]
    slope = dy / dT

    T_mid     = (Tk[1:] + Tk[:-1]) / 2.0
    mask_post = T_mid > ogtK
    if mask_post.sum() == 0:
        return torch.tensor(0.0, device=y_hat.device)

    post_slopes = slope[mask_post]
    return torch.mean(F.relu(post_slopes) ** 2)


def compute_total_loss(y_pred, y_true, Tk, ogtK, loss_fn):
    """
    Combine the supervised data loss with the three PINN constraint losses.

    Non-negativity is NOT included here because it is enforced as a hard
    constraint via softplus in the model output layer.

    Returns
    -------
    (L_total, L_data, L_arrhenius, L_peak_height, L_tail_decline)
    """
    loss_data = loss_fn(y_pred, y_true)
    loss_arr  = compute_arrhenius_loss(y_hat=y_pred, Tk=Tk, ogtK=ogtK)
    loss_peak = compute_peak_height_loss(y_hat=y_pred, Tk=Tk, ogtK=ogtK)
    loss_tail = compute_tail_decline_loss(y_hat=y_pred, Tk=Tk, ogtK=ogtK)

    loss_total = (loss_data
                  + LAMBDA_ARR  * loss_arr
                  + LAMBDA_PEAK * loss_peak
                  + LAMBDA_TAIL * loss_tail)

    return loss_total, loss_data, loss_arr, loss_peak, loss_tail


# ══════════════════════════════════════════════════════════════
# Neural-network modules
# ══════════════════════════════════════════════════════════════

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
            f"emb_len={emb_len} must be divisible by n_patches={n_patches}"
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
        x        = self.patch_mlp(emb_vec.view(B, self.n_patches, self.patch_dim))
        temp_tok = self.temp_proj(tfeat).unsqueeze(1)
        x        = torch.cat([temp_tok, x], dim=1).transpose(0, 1)
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
            nn.Linear(hidden, 64),         nn.GELU(),
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
            # Hard gate: residual contribution forced negative post-Topt
            dres = torch.where(Tc > ToptC, -F.softplus(dres_raw), dres_raw)
        else:
            dres = dres_raw
        dy = dbase + dres
        x  = (Tc - ToptC) / (E + 1e-8)
        dy = torch.where((x > 1.0) & (y_in <= 0.0), torch.zeros_like(dy), dy)
        return dy


class UDEModel_PINN(nn.Module):
    """
    UDE model with PINN hard non-negativity constraint.

    Key difference from group4: replaces torch.clamp(min=0) with F.softplus so
    that y_hat >= 0 everywhere while remaining differentiable, as required for
    the PINN constraint losses.
    """
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

        # ── Hard non-negativity via softplus (PINN constraint, replaces clamp) ──
        y_pred = F.softplus(traj.squeeze(-1))

        t_norm    = (Tk_vec - t_mean_k) / (t_std_k + 1e-8)
        z_use     = z.detach() if self.odefunc.detach_z else z
        resid_seq = self.odefunc.residual(t_norm, y_pred, z_use.expand(Tk_vec.shape[0], -1))
        return y_pred, resid_seq, (Pmax, ToptC, E)


# ══════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════

def load_data():
    df_raw = pd.read_csv(DATA_CSV, low_memory=False)
    for c in [COL_ID, COL_SPECIES, COL_TEMP, COL_KINGDOM, COL_OGT]:
        assert c in df_raw.columns, f"Missing required column: {c}"
    assert COL_Y in df_raw.columns

    df_raw[COL_KINGDOM] = df_raw[COL_KINGDOM].astype(str)
    df_raw = df_raw[df_raw[COL_KINGDOM].isin(KINGDOM_KEEP)].copy()

    df_raw[COL_TEMP] = pd.to_numeric(df_raw[COL_TEMP], errors="coerce")
    df_raw[COL_Y]    = pd.to_numeric(df_raw[COL_Y],    errors="coerce").fillna(0.0)
    df_raw[COL_OGT]  = pd.to_numeric(df_raw[COL_OGT],  errors="coerce")

    ogt_med = df_raw[COL_OGT].median()
    if pd.notna(ogt_med) and ogt_med > 150:
        print("[INFO] OGT appears to be in Kelvin, converting: OGT -= 273.15")
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
    assert len(esm_cols) > 0, "No ESM embedding columns found"
    print(f"ESM dims = {len(esm_cols)}")

    before = len(df_raw)
    df_raw = df_raw.dropna(subset=esm_cols, how="any").reset_index(drop=True)
    print(f"Dropped ESM-null rows: {before} -> {len(df_raw)}")

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
                COL_OGT_SIM: base_map[tid] + deterministic_noise_pm5(tid, SIM_SEED),
            } for tid in miss_ids]
            sim_df = pd.concat([sim_df, pd.DataFrame(new_rows)], ignore_index=True)
            sim_df.drop_duplicates(subset=[COL_ID], keep="first", inplace=True)
            sim_df.to_csv(SIM_FILE, index=False)
    else:
        rows = []
        for _, r in curve_ogt_base.sort_values(COL_ID).iterrows():
            tid   = str(r[COL_ID])
            ogt0  = float(r[COL_OGT])
            delta = deterministic_noise_pm5(tid, SIM_SEED)
            rows.append({COL_ID: tid, "OGT_orig_C": ogt0, "delta_C": delta,
                         COL_OGT_SIM: ogt0 + delta})
        sim_df = pd.DataFrame(rows)
        sim_df.to_csv(SIM_FILE, index=False)
        print(f"[INFO] Created OGT simulator file: {SIM_FILE}")

    df = df.merge(sim_df[[COL_ID, COL_OGT_SIM]], on=COL_ID, how="left")
    assert df[COL_OGT_SIM].isna().sum() == 0

    print(f"Data ready: {len(df)} rows | {df[COL_ID].nunique()} curves | "
          f"{df[COL_SPECIES].nunique()} species")
    return df, esm_cols


# ══════════════════════════════════════════════════════════════
# Training (full dataset, no cross-validation)
# ══════════════════════════════════════════════════════════════

def train_all(df, esm_cols):
    print("\n=== Group IV PINN -- UTPC + Biophysical Constraints | Full Training ===")
    print(f"PINN weights: lambda_arr={LAMBDA_ARR}  lambda_peak={LAMBDA_PEAK}  "
          f"lambda_tail={LAMBDA_TAIL}")
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
    model     = UDEModel_PINN(encoder, head, residual).to(device)

    theta_params = list(encoder.parameters()) + list(head.parameters())
    resid_params = list(residual.parameters())
    opt_theta = torch.optim.Adam(theta_params, lr=LR_THETA, weight_decay=WEIGHT_DECAY)
    opt_resid = torch.optim.Adam(resid_params, lr=LR_RESID, weight_decay=WEIGHT_DECAY)
    loss_fn   = nn.SmoothL1Loss()

    log_rows   = []
    ep_counter = {"v": 0}

    def run_epoch(train_theta: bool, train_resid: bool, tag: str):
        ep_counter["v"] += 1
        model.train()
        set_requires_grad(encoder,  train_theta)
        set_requires_grad(head,     train_theta)
        set_requires_grad(residual, train_resid)

        acc  = dict(data=0., arr=0., peak=0., tail=0., reg=0., total=0.)
        ncur = 0

        for tid in random.sample(list(all_curves), len(all_curves)):
            C    = all_curves[tid]
            Tk   = torch.tensor(C["Tk"],      dtype=torch.float32, device=device)
            ogtK = torch.tensor(C["ogtK"],    dtype=torch.float32, device=device)
            y_ts = torch.tensor(C["y_shape"], dtype=torch.float32, device=device)
            emb  = torch.tensor(C["emb"],     dtype=torch.float32, device=device)

            opt_theta.zero_grad(set_to_none=True)
            opt_resid.zero_grad(set_to_none=True)

            y_raw, resid_seq, _ = model.forward_curve(emb, Tk, ogtK, t_mean_k, t_std_k)

            # Normalise to shape with peak ≈ 1.
            # Non-negativity is guaranteed by softplus in forward_curve,
            # so division by a positive max is safe.
            y_max  = y_raw.max().clamp(min=1e-8)
            y_pred = y_raw / y_max

            # PINN constraint losses + data loss
            loss_total, loss_data, loss_arr, loss_peak, loss_tail = compute_total_loss(
                y_pred, y_ts, Tk, ogtK, loss_fn)

            # Residual regularisation (only when training the residual net)
            loss_reg = torch.mean(resid_seq ** 2)
            if train_resid:
                loss_total = loss_total + RES_LAMBDA * loss_reg

            loss_total.backward()

            if train_theta:
                torch.nn.utils.clip_grad_norm_(theta_params, CLIP_NORM)
                opt_theta.step()
            if train_resid:
                torch.nn.utils.clip_grad_norm_(resid_params, CLIP_NORM)
                opt_resid.step()

            acc["data"]  += float(loss_data.item())
            acc["arr"]   += float(loss_arr.item())
            acc["peak"]  += float(loss_peak.item())
            acc["tail"]  += float(loss_tail.item())
            acc["reg"]   += float(loss_reg.item())
            acc["total"] += float(loss_total.item())
            ncur         += 1

        n    = max(1, ncur)
        avgs = {k: v / n for k, v in acc.items()}
        print(f"  {tag:32s} | total {avgs['total']:.5f}  "
              f"(data {avgs['data']:.5f} | arr {avgs['arr']:.5f} | "
              f"peak {avgs['peak']:.5f} | tail {avgs['tail']:.5f} | "
              f"reg {avgs['reg']:.5f})")

        log_rows.append({
            "ep":                ep_counter["v"],
            "tag":               tag,
            "loss_data":         avgs["data"],
            "loss_arrhenius":    avgs["arr"],
            "loss_peak_height":  avgs["peak"],
            "loss_tail_decline": avgs["tail"],
            "loss_reg":          avgs["reg"],
            "loss_total":        avgs["total"],
        })

    # ── Phase 1: warmup — encoder + head only ──
    for ep in range(1, WARMUP_EPOCHS + 1):
        run_epoch(True, False, f"[Warmup theta] Ep {ep:02d}")

    # ── Phase 2: alternating optimisation ──
    for cyc in range(1, ALT_CYCLES + 1):
        for ep in range(1, ALT_EPOCHS_THETA + 1):
            run_epoch(True,  False, f"[Alt{cyc} theta] Ep {ep:02d}")
        for ep in range(1, ALT_EPOCHS_RESID + 1):
            run_epoch(False, True,  f"[Alt{cyc} resid] Ep {ep:02d}")

    # ── Phase 3: joint fine-tuning at lower LR ──
    for g in opt_theta.param_groups: g["lr"] = LR_THETA * 0.3
    for g in opt_resid.param_groups: g["lr"] = LR_RESID * 0.3
    for ep in range(1, JOINT_EPOCHS + 1):
        run_epoch(True, True, f"[Joint] Ep {ep:02d}")

    elapsed = time.time() - t0
    print(f"\nTraining complete: {elapsed:.1f}s")

    # ── Save training log (Train/) ──
    log_df   = pd.DataFrame(log_rows)
    log_path = TRAIN_DIR / "group4_pinn_training_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Training log: {log_path}")

    # ── Loss-component plots (Train/) ──
    loss_cols = [
        ("loss_total",        "Total Loss",         "tab:blue"),
        ("loss_data",         "Data Loss (SmoothL1)","tab:orange"),
        ("loss_arrhenius",    "Arrhenius Loss",      "tab:green"),
        ("loss_peak_height",  "Peak Height Loss",    "tab:red"),
        ("loss_tail_decline", "Tail Decline Loss",   "tab:purple"),
        ("loss_reg",          "Residual Reg Loss",   "tab:gray"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (col, title, color) in zip(axes.flatten(), loss_cols):
        ax.plot(log_df["ep"], log_df[col], linewidth=1.5, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        f"Group IV PINN -- Training Loss Components\n"
        f"lam_arr={LAMBDA_ARR}  lam_peak={LAMBDA_PEAK}  lam_tail={LAMBDA_TAIL}",
        fontsize=12
    )
    plt.tight_layout()
    plot_path = TRAIN_DIR / "group4_pinn_training_loss.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Loss plot:     {plot_path}")

    # ── Summary statistics (Train/) ──
    final_ep = log_df.iloc[-1]
    summary = {
        "total_epochs":      int(ep_counter["v"]),
        "elapsed_seconds":   round(elapsed, 1),
        "lambda_arr":        LAMBDA_ARR,
        "lambda_peak":       LAMBDA_PEAK,
        "lambda_tail":       LAMBDA_TAIL,
        "final_loss_total":  round(float(final_ep["loss_total"]),        6),
        "final_loss_data":   round(float(final_ep["loss_data"]),         6),
        "final_loss_arr":    round(float(final_ep["loss_arrhenius"]),    6),
        "final_loss_peak":   round(float(final_ep["loss_peak_height"]),  6),
        "final_loss_tail":   round(float(final_ep["loss_tail_decline"]), 6),
        "final_loss_reg":    round(float(final_ep["loss_reg"]),          6),
    }
    import json
    summary_path = TRAIN_DIR / "group4_pinn_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary:       {summary_path}")

    return model, emb_scaler, t_mean_k, t_std_k, EMB_LEN, n_patches


# ══════════════════════════════════════════════════════════════
# Save model to results/
# ══════════════════════════════════════════════════════════════

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
            "ATTN_DIM":    ATTN_DIM, "N_HEADS":   N_HEADS,  "N_LAYERS":  N_LAYERS,
            "DROPOUT":     DROPOUT,  "Z_DIM":     Z_DIM,    "Y_CLIP":    Y_CLIP,
            "HARD_NO_INCREASE_AFTER_OGT": HARD_NO_INCREASE_AFTER_OGT,
            "LAMBDA_ARR":  LAMBDA_ARR,
            "LAMBDA_PEAK": LAMBDA_PEAK,
            "LAMBDA_TAIL": LAMBDA_TAIL,
            "nonneg_mode": "softplus",   # marks that softplus was used
        },
    }, CHECKPOINT_PATH)
    print(f"Model saved:   {CHECKPOINT_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(emb_scaler, f)
    print(f"Scaler saved:  {SCALER_PATH}")


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df, esm_cols = load_data()
    model, emb_scaler, t_mean_k, t_std_k, EMB_LEN, n_patches = train_all(df, esm_cols)
    save_model(model, emb_scaler, t_mean_k, t_std_k, EMB_LEN, n_patches, esm_cols)
    print(f"\nDone.")
    print(f"  Model / scaler  -> {RESULTS_DIR}")
    print(f"  Training logs   -> {TRAIN_DIR}")
