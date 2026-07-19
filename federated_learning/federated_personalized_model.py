import json
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import copy

from federated_learning.paths import (
    FL_DIR,
    MCP_HORM_PATH as _MCP,
    NHANES_DEMO,
    NHANES_TST,
    RESULTS_DIR,
    require_mcphases,
    rounds as pfl_rounds,
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

P_TST_PATH = str(NHANES_TST)
DEMO_PATH = str(NHANES_DEMO)
MCP_HORM_PATH = str(require_mcphases()) if _MCP.exists() else str(_MCP)
OUTPUT_DIR = str(FL_DIR)
PLOT_PATH = str(FL_DIR / "federated_evaluation_comparison.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hormone column lists
NHANES_HORM_COLS = [
    'LBX17H', 'LBXAND', 'LBXAMH', 'LBXEST', 'LBXESO', 
    'LBXES1', 'LBXFSH', 'LBXLUH', 'LBXPG4', 'LBXSHBG'
]
MCP_HORM_COLS = ['lh', 'estrogen', 'pdg']

# KNN Imputer in NumPy (identical to baseline files)
def knn_impute_numpy(X, k=5):
    n_samples, n_features = X.shape
    col_means = np.nanmean(X, axis=0)
    col_means[np.isnan(col_means)] = 0.0
    
    nan_mask = np.isnan(X)
    X_norm = X.copy()
    col_stds = np.nanstd(X, axis=0)
    col_stds[col_stds == 0] = 1.0
    X_norm = (X_norm - col_means) / col_stds
    
    for i in range(n_samples):
        nan_cols = np.where(nan_mask[i])[0]
        if len(nan_cols) == 0:
            continue
        non_nan_cols = np.where(~nan_mask[i])[0]
        if len(non_nan_cols) == 0:
            X[i] = col_means
            continue
            
        diffs = X_norm[:, non_nan_cols] - X_norm[i, non_nan_cols]
        other_nans = np.isnan(diffs)
        diffs[other_nans] = 0.0
        
        sq_dist = np.sum(diffs**2, axis=1)
        n_shared = np.sum(~other_nans, axis=1)
        n_shared[n_shared == 0] = 1
        dist = sq_dist / n_shared
        dist[i] = np.inf
        
        for col in nan_cols:
            has_val = ~nan_mask[:, col]
            if not np.any(has_val):
                X[i, col] = col_means[col]
                continue
            valid_dists = dist[has_val]
            valid_indices = np.where(has_val)[0]
            nearest_local_indices = np.argsort(valid_dists)[:k]
            nearest_global_indices = valid_indices[nearest_local_indices]
            X[i, col] = np.mean(X[nearest_global_indices, col])
            
    return X

# 1. Models Definition

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim=8):
        super(ProjectionLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class SharedEncoder(nn.Module):
    def __init__(self, dim=8):
        super(SharedEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)

class DecisionHead(nn.Module):
    def __init__(self, dim=8):
        super(DecisionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

# Full Personalized Client Model (Combines Projection, Shared Encoder, and Decision Head)
class PersonalizedClientModel(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super(PersonalizedClientModel, self).__init__()
        self.proj = ProjectionLayer(input_dim, latent_dim)
        self.encoder = SharedEncoder(latent_dim)
        self.head = DecisionHead(latent_dim)
        
    def forward(self, x):
        features = self.proj(x)
        latent = self.encoder(features)
        out = self.head(latent)
        return out

# Helper to train a model locally
def train_local_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    return epoch_loss / len(loader.dataset)

# Helper to evaluate a model
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            preds = model(batch_x).squeeze(1).numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.squeeze(1).numpy())
            
    y_true = np.array(all_targets)
    y_pred_prob = np.array(all_preds)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1
    }

def average_state_dicts(state_dicts):
    avg_dict = copy.deepcopy(state_dicts[0])
    for key in avg_dict.keys():
        for i in range(1, len(state_dicts)):
            avg_dict[key] += state_dicts[i][key]
        avg_dict[key] = avg_dict[key] / float(len(state_dicts))
    return avg_dict

def main():
    print("=== Step 1: Loading & Preprocessing NHANES ===")
    df_tst = pd.read_csv(P_TST_PATH)
    df_demo = pd.read_sas(DEMO_PATH, format='xport')[['SEQN', 'RIAGENDR']]
    df_nhanes = pd.merge(df_tst, df_demo, on='SEQN').dropna(subset=NHANES_HORM_COLS, how='all')
    
    # Split by Gender
    df_nh_female = df_nhanes[df_nhanes['RIAGENDR'] == 2].copy()
    df_nh_male = df_nhanes[df_nhanes['RIAGENDR'] == 1].copy()
    
    # Log transform
    for col in NHANES_HORM_COLS:
        df_nh_female[col] = np.log1p(df_nh_female[col])
        df_nh_male[col] = np.log1p(df_nh_male[col])
        
    # Impute missing values using KNN
    print("Imputing NHANES Females...")
    X_nh_female = knn_impute_numpy(df_nh_female[NHANES_HORM_COLS].values, k=5)
    print("Imputing NHANES Males...")
    X_nh_male = knn_impute_numpy(df_nh_male[NHANES_HORM_COLS].values, k=5)
    
    # Define Target deficiency labels (low Progesterone < 20th percentile)
    p_pg4_f = np.percentile(X_nh_female[:, 8], 20)
    p_pg4_m = np.percentile(X_nh_male[:, 8], 20)
    
    y_nh_female = (X_nh_female[:, 8] < p_pg4_f).astype(int)
    y_nh_male = (X_nh_male[:, 8] < p_pg4_m).astype(int)
    
    # --- Client 1: Balanced NHANES ---
    print("\nSetting up Client 1 (Balanced NHANES: 50% Male, 50% Female)...")
    idx_f1 = np.random.choice(len(X_nh_female), 750, replace=False)
    idx_m1 = np.random.choice(len(X_nh_male), 750, replace=False)
    
    X_c1 = np.vstack([X_nh_female[idx_f1], X_nh_male[idx_m1]])
    y_c1 = np.concatenate([y_nh_female[idx_f1], y_nh_male[idx_m1]])
    
    # --- Client 2: Male-skewed NHANES ---
    print("Setting up Client 2 (Male-skewed NHANES: 90% Male, 10% Female)...")
    # Exclude indices used by Client 1 to prevent data leakage
    f_rem_idx = np.setdiff1d(np.arange(len(X_nh_female)), idx_f1)
    m_rem_idx = np.setdiff1d(np.arange(len(X_nh_male)), idx_m1)
    
    idx_f2 = np.random.choice(f_rem_idx, 150, replace=False)
    idx_m2 = np.random.choice(m_rem_idx, 1350, replace=False)
    
    X_c2 = np.vstack([X_nh_female[idx_f2], X_nh_male[idx_m2]])
    y_c2 = np.concatenate([y_nh_female[idx_f2], y_nh_male[idx_m2]])
    
    # === Step 2: Loading & Preprocessing McPhases ===
    print("\n=== Step 3: Loading & Preprocessing McPhases (Client 3) ===")
    df_mcp = pd.read_csv(MCP_HORM_PATH).dropna(subset=MCP_HORM_COLS, how='all')
    
    # Log transform
    for col in MCP_HORM_COLS:
        df_mcp[col] = np.log1p(df_mcp[col])
        
    print("Imputing McPhases cyclic data...")
    X_mcp = knn_impute_numpy(df_mcp[MCP_HORM_COLS].values, k=5)
    
    # Target label: low PdG (Progesterone metabolite) < 20th percentile
    p_pdg = np.percentile(X_mcp[:, 2], 20)
    y_mcp = (X_mcp[:, 2] < p_pdg).astype(int)
    
    # Take a subset of 2000 points for Client 3
    idx_c3 = np.random.choice(len(X_mcp), 2000, replace=False)
    X_c3 = X_mcp[idx_c3]
    y_c3 = y_mcp[idx_c3]
    
    # === Step 3: Train / Test Splits & Scaling ===
    def get_train_test_split(X, y, train_ratio=0.8):
        n = len(X)
        indices = np.random.permutation(n)
        split = int(n * train_ratio)
        train_idx, test_idx = indices[:split], indices[split:]
        
        # Scale locally
        mean = np.mean(X[train_idx], axis=0)
        std = np.std(X[train_idx], axis=0)
        std[std == 0] = 1.0
        
        X_scaled = (X - mean) / std
        return X_scaled[train_idx], y[train_idx], X_scaled[test_idx], y[test_idx]

    X_c1_tr, y_c1_tr, X_c1_te, y_c1_te = get_train_test_split(X_c1, y_c1)
    X_c2_tr, y_c2_tr, X_c2_te, y_c2_te = get_train_test_split(X_c2, y_c2)
    X_c3_tr, y_c3_tr, X_c3_te, y_c3_te = get_train_test_split(X_c3, y_c3)
    
    print(f"\nFinal Train/Test Shapes:")
    print(f"Client 1 (Balanced NHANES)  - Train: {X_c1_tr.shape}, Test: {X_c1_te.shape}")
    print(f"Client 2 (Male-skewed NH)   - Train: {X_c2_tr.shape}, Test: {X_c2_te.shape}")
    print(f"Client 3 (McPhases Cyclic)  - Train: {X_c3_tr.shape}, Test: {X_c3_te.shape}")
    
    # Create DataLoaders
    def build_loader(X, y, batch_size=64, shuffle=True):
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
    c1_tr_loader = build_loader(X_c1_tr, y_c1_tr)
    c1_te_loader = build_loader(X_c1_te, y_c1_te, shuffle=False)
    
    c2_tr_loader = build_loader(X_c2_tr, y_c2_tr)
    c2_te_loader = build_loader(X_c2_te, y_c2_te, shuffle=False)
    
    c3_tr_loader = build_loader(X_c3_tr, y_c3_tr)
    c3_te_loader = build_loader(X_c3_te, y_c3_te, shuffle=False)
    
    # Setup parameters
    lr = 0.003
    epochs_per_round = 3
    rounds = pfl_rounds(30)
    print(f"Running multi-site pFL with rounds={rounds}")
    criterion = nn.BCELoss()
    
    # Dictionary to keep evaluation metrics
    results = {
        "Client 1": {"Local Only": {}, "Centralized/FedAvg": {}, "Personalized FL": {}},
        "Client 2": {"Local Only": {}, "Centralized/FedAvg": {}, "Personalized FL": {}},
        "Client 3": {"Local Only": {}, "Centralized/FedAvg": {}, "Personalized FL": {}}
    }
    
    # =========================================================================
    # CONFIGURATION 1: LOCAL ONLY (NO PARTNERSHIP / AGGREGATION)
    # =========================================================================
    print("\n--- Training Paradigm 1: Local Only Models ---")
    local_m1 = PersonalizedClientModel(input_dim=10)
    local_m2 = PersonalizedClientModel(input_dim=10)
    local_m3 = PersonalizedClientModel(input_dim=3)
    
    opt1 = optim.Adam(local_m1.parameters(), lr=lr)
    opt2 = optim.Adam(local_m2.parameters(), lr=lr)
    opt3 = optim.Adam(local_m3.parameters(), lr=lr)
    
    total_local_epochs = rounds * epochs_per_round
    for epoch in range(total_local_epochs):
        train_local_epoch(local_m1, c1_tr_loader, opt1, criterion)
        train_local_epoch(local_m2, c2_tr_loader, opt2, criterion)
        train_local_epoch(local_m3, c3_tr_loader, opt3, criterion)
        
    results["Client 1"]["Local Only"] = evaluate_model(local_m1, c1_te_loader)
    results["Client 2"]["Local Only"] = evaluate_model(local_m2, c2_te_loader)
    results["Client 3"]["Local Only"] = evaluate_model(local_m3, c3_te_loader)
    
    # =========================================================================
    # CONFIGURATION 2: CENTRALIZED / STANDARD FEDAVG (WITH DIM ALIGNMENT)
    # =========================================================================
    print("\n--- Training Paradigm 2: Standard FedAvg (Encoder & Head Shared) ---")
    
    # Create persistent client models
    fedavg_m1 = PersonalizedClientModel(input_dim=10)
    fedavg_m2 = PersonalizedClientModel(input_dim=10)
    fedavg_m3 = PersonalizedClientModel(input_dim=3)
    
    # Persistent optimizers
    opt_fedavg1 = optim.Adam(fedavg_m1.parameters(), lr=lr)
    opt_fedavg2 = optim.Adam(fedavg_m2.parameters(), lr=lr)
    opt_fedavg3 = optim.Adam(fedavg_m3.parameters(), lr=lr)
    
    for r in range(rounds):
        # 1. Local update epochs
        for _ in range(epochs_per_round):
            train_local_epoch(fedavg_m1, c1_tr_loader, opt_fedavg1, criterion)
            train_local_epoch(fedavg_m2, c2_tr_loader, opt_fedavg2, criterion)
            train_local_epoch(fedavg_m3, c3_tr_loader, opt_fedavg3, criterion)
            
        # 2. Aggregation: compute averages for Encoder and Head
        avg_enc_state = average_state_dicts([
            fedavg_m1.encoder.state_dict(),
            fedavg_m2.encoder.state_dict(),
            fedavg_m3.encoder.state_dict()
        ])
        avg_head_state = average_state_dicts([
            fedavg_m1.head.state_dict(),
            fedavg_m2.head.state_dict(),
            fedavg_m3.head.state_dict()
        ])
        
        # 3. Synchronize aggregated parameters back to persistent client models
        fedavg_m1.encoder.load_state_dict(avg_enc_state)
        fedavg_m2.encoder.load_state_dict(avg_enc_state)
        fedavg_m3.encoder.load_state_dict(avg_enc_state)
        
        fedavg_m1.head.load_state_dict(avg_head_state)
        fedavg_m2.head.load_state_dict(avg_head_state)
        fedavg_m3.head.load_state_dict(avg_head_state)
        
    results["Client 1"]["Centralized/FedAvg"] = evaluate_model(fedavg_m1, c1_te_loader)
    results["Client 2"]["Centralized/FedAvg"] = evaluate_model(fedavg_m2, c2_te_loader)
    results["Client 3"]["Centralized/FedAvg"] = evaluate_model(fedavg_m3, c3_te_loader)

    # =========================================================================
    # CONFIGURATION 3: PERSONALIZED FL (pFL / FEDPER - SHARED ENCODER ONLY)
    # =========================================================================
    print("\n--- Training Paradigm 3: Personalized Federated Learning (FedPer) ---")
    
    # Create persistent client models
    pfl_m1 = PersonalizedClientModel(input_dim=10)
    pfl_m2 = PersonalizedClientModel(input_dim=10)
    pfl_m3 = PersonalizedClientModel(input_dim=3)
    
    # Persistent optimizers
    opt_pfl1 = optim.Adam(pfl_m1.parameters(), lr=lr)
    opt_pfl2 = optim.Adam(pfl_m2.parameters(), lr=lr)
    opt_pfl3 = optim.Adam(pfl_m3.parameters(), lr=lr)
    
    for r in range(rounds):
        # 1. Local update epochs
        for _ in range(epochs_per_round):
            train_local_epoch(pfl_m1, c1_tr_loader, opt_pfl1, criterion)
            train_local_epoch(pfl_m2, c2_tr_loader, opt_pfl2, criterion)
            train_local_epoch(pfl_m3, c3_tr_loader, opt_pfl3, criterion)
            
        # 2. Aggregation: compute average ONLY for the shared Encoder
        avg_enc_state = average_state_dicts([
            pfl_m1.encoder.state_dict(),
            pfl_m2.encoder.state_dict(),
            pfl_m3.encoder.state_dict()
        ])
        
        # 3. Synchronize aggregated encoder back to persistent client models
        # (The projection layers and prediction heads remain fully personalized and keep their local gradients)
        pfl_m1.encoder.load_state_dict(avg_enc_state)
        pfl_m2.encoder.load_state_dict(avg_enc_state)
        pfl_m3.encoder.load_state_dict(avg_enc_state)
        
    results["Client 1"]["Personalized FL"] = evaluate_model(pfl_m1, c1_te_loader)
    results["Client 2"]["Personalized FL"] = evaluate_model(pfl_m2, c2_te_loader)
    results["Client 3"]["Personalized FL"] = evaluate_model(pfl_m3, c3_te_loader)
    
    # =========================================================================
    # PRINT RESULTS TABLE
    # =========================================================================
    print("\n" + "="*90)
    print("                      FEDERATED SIMULATION EVALUATION COMPARISON")
    print("="*90)
    print(f"{'Client / Metric':<28} | {'Local Only':<15} | {'Centralized / FedAvg':<22} | {'Personalized FL (pFL)':<22}")
    print("-"*90)
    
    for client in ["Client 1", "Client 2", "Client 3"]:
        print(f"--- {client} ---")
        for metric in ["accuracy", "sensitivity", "specificity", "f1"]:
            val_local = results[client]["Local Only"][metric]
            val_fedavg = results[client]["Centralized/FedAvg"][metric]
            val_pfl = results[client]["Personalized FL"][metric]
            print(f"  {metric.capitalize():<24} | {val_local:<15.4f} | {val_fedavg:<22.4f} | {val_pfl:<22.4f}")
        print("-"*90)
        
    # =========================================================================
    # PLOT COMPARISON GRAPH
    # =========================================================================
    metrics = ["accuracy", "sensitivity", "specificity", "f1"]
    metric_labels = ["Accuracy", "Sensitivity", "Specificity", "F1 Score"]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(metrics))
    width = 0.25
    
    colors = {
        "Local Only": "indianred",
        "Centralized/FedAvg": "dodgerblue",
        "Personalized FL": "seagreen"
    }
    
    for idx, client in enumerate(["Client 1", "Client 2", "Client 3"]):
        local_vals = [results[client]["Local Only"][m] for m in metrics]
        fedavg_vals = [results[client]["Centralized/FedAvg"][m] for m in metrics]
        pfl_vals = [results[client]["Personalized FL"][m] for m in metrics]
        
        axes[idx].bar(x - width, local_vals, width, label='Local Only', color=colors["Local Only"])
        axes[idx].bar(x, fedavg_vals, width, label='Centralized/FedAvg', color=colors["Centralized/FedAvg"])
        axes[idx].bar(x + width, pfl_vals, width, label='Personalized FL (pFL)', color=colors["Personalized FL"])
        
        axes[idx].set_title(f"{client} Performance")
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(metric_labels)
        axes[idx].set_ylim(0, 1.1)
        axes[idx].grid(True, linestyle='--', alpha=0.3)
        if idx == 0:
            axes[idx].set_ylabel("Score")
            axes[idx].legend()
            
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"\nFederated simulation evaluation plot successfully saved to: {PLOT_PATH}")

if __name__ == "__main__":
    main()
