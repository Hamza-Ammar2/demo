import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import copy

from federated_learning.paths import FL_DIR, RESULTS_DIR, require_mcphases, rounds as pfl_rounds

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Paths (repo-relative — see federated_learning/paths.py)
MCP_HORM_PATH = str(require_mcphases())
OUTPUT_DIR = str(FL_DIR)
PLOT_PATH = str(FL_DIR / "multi_symptom_comparison.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature columns
HORM_COLS = ['lh', 'estrogen', 'pdg']
SYMP_COLS = ['headaches', 'sorebreasts', 'fatigue', 'sleepissue', 'moodswing', 'stress', 'bloating']

# Symptom ordinal mapping dictionary
SYMPTOM_MAP = {
    'Not at all': 0,
    'Very Low/Little': 1,
    'Low': 2,
    'Moderate': 3,
    'High': 4,
    'Very High': 5
}

def map_symptom(val):
    if pd.isna(val):
        return 0
    return SYMPTOM_MAP.get(str(val).strip(), 0)

# 1. Models Definition (Temporal GRU Setup for 10-dimensional inputs)

class GRUProjection(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16, output_dim=8):
        super(GRUProjection, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.ln = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        # Take the final output step (final sequence hidden state)
        final_state = out[:, -1, :]
        features = self.fc(final_state)
        return self.ln(features)

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
            nn.Linear(8, 1)  # Raw logits output
        )
    def forward(self, x):
        return self.net(x)

class PersonalizedClientModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=16, latent_dim=8):
        super(PersonalizedClientModel, self).__init__()
        self.proj = GRUProjection(input_dim, hidden_dim, latent_dim)
        self.encoder = SharedEncoder(latent_dim)
        self.head = DecisionHead(latent_dim)
        
    def forward(self, x):
        features = self.proj(x)
        latent = self.encoder(features)
        out = self.head(latent)
        return out

# Helper to train local epoch
def train_local_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    return epoch_loss / len(loader.dataset)

# Helper to evaluate model
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            preds = torch.sigmoid(logits).squeeze(1).numpy()
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
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else np.nan
    
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
    print("=== Step 1: Loading & Cleaning McPhases ===")
    df = pd.read_csv(MCP_HORM_PATH)
    
    # Map cramps symptoms: Moderate, High, Very High to 1, else 0
    df['cramps_binary'] = df['cramps'].apply(lambda val: 1 if val in ['Moderate', 'High', 'Very High'] else 0)
    df = df.dropna(subset=HORM_COLS, how='all')
    
    clients_data = []
    unique_subject_ids = df['id'].unique()
    
    for sub_id in unique_subject_ids:
        sub_df = df[df['id'] == sub_id].sort_values(by='day_in_study').copy()
        
        # Map symptom logs to ordinals
        for col in SYMP_COLS:
            sub_df[col] = sub_df[col].apply(map_symptom)
            
        # Log-transform hormones
        for col in HORM_COLS:
            sub_df[col] = np.log1p(sub_df[col])
            
        # Interpolate and pad locally
        sub_df[HORM_COLS] = sub_df[HORM_COLS].interpolate(method='linear').ffill().bfill()
        sub_df = sub_df.dropna(subset=HORM_COLS)
        
        if len(sub_df) < 30:
            continue
            
        X = sub_df[HORM_COLS + SYMP_COLS].values
        y = sub_df['cramps_binary'].values
        
        # Build sliding windows of size W = 5
        W = 5
        X_seq = []
        y_seq = []
        for t in range(W - 1, len(X)):
            X_seq.append(X[t - W + 1 : t + 1])
            y_seq.append(y[t])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Minimum sequences check
        if len(X_seq) < 25:
            continue
            
        # Split randomly
        n_samples = len(X_seq)
        shuffled_idx = np.random.permutation(n_samples)
        split = int(0.8 * n_samples)
        
        train_idx = shuffled_idx[:split]
        test_idx = shuffled_idx[split:]
        
        X_tr, y_tr = X_seq[train_idx], y_seq[train_idx]
        X_te, y_te = X_seq[test_idx], y_seq[test_idx]
        
        # Balance check
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
            
        # Standardize features locally (along input channels)
        N_tr, seq_len, channels = X_tr.shape
        X_tr_2d = X_tr.reshape(-1, channels)
        mean = np.mean(X_tr_2d, axis=0)
        std = np.std(X_tr_2d, axis=0)
        std[std == 0] = 1.0
        
        X_tr_scaled = ((X_tr.reshape(-1, channels) - mean) / std).reshape(N_tr, seq_len, channels)
        X_te_scaled = ((X_te.reshape(-1, channels) - mean) / std).reshape(X_te.shape[0], seq_len, channels)
        
        # Compute pos_weight for local loss
        num_neg = np.sum(y_tr == 0)
        num_pos = np.sum(y_tr == 1)
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32)
        client_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        clients_data.append({
            "id": sub_id,
            "X_train": X_tr_scaled,
            "y_train": y_tr,
            "X_test": X_te_scaled,
            "y_test": y_te,
            "criterion": client_criterion
        })
        
    num_clients = len(clients_data)
    print(f"\nFiltered McPhases Cohort: Selected {num_clients} unique patient clients with multi-symptom sequential datasets.")
    if num_clients == 0:
        print("Error: No patients matched the selection criteria. Exiting.")
        return
        
    # Helper to construct PyTorch loader
    def build_loader(X, y, batch_size=8, shuffle=True):
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(tensor_x, tensor_y)
        drop_last = shuffle and (len(dataset) > batch_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        
    for client in clients_data:
        client["train_loader"] = build_loader(client["X_train"], client["y_train"])
        client["test_loader"] = build_loader(client["X_test"], client["y_test"], shuffle=False)
        
    # Hyperparameters (override rounds with PFL_ROUNDS=3 for a quick smoke test)
    lr = 0.003
    epochs_per_round = 3
    rounds = pfl_rounds(30)
    print(f"Running multi-symptom pFL with rounds={rounds}")
    
    eval_paradigms = ["Local Only", "Centralized/FedAvg", "Personalized FL"]
    results = {p: {metric: [] for metric in ["accuracy", "sensitivity", "specificity", "f1"]} for p in eval_paradigms}
    
    # =========================================================================
    # PARADIGM 1: LOCAL ONLY (NO PARAMETER SHARING)
    # =========================================================================
    print("\n--- Training Paradigm 1: Local Only Patients ---")
    local_models = [PersonalizedClientModel() for _ in range(num_clients)]
    optimizers = [optim.Adam(local_models[i].parameters(), lr=lr) for i in range(num_clients)]
    
    total_local_epochs = rounds * epochs_per_round
    for epoch in range(total_local_epochs):
        for i, client in enumerate(clients_data):
            train_local_epoch(local_models[i], client["train_loader"], optimizers[i], client["criterion"])
            
    for i, client in enumerate(clients_data):
        metrics = evaluate_model(local_models[i], client["test_loader"])
        for m in metrics.keys():
            results["Local Only"][m].append(metrics[m])
            
    # =========================================================================
    # PARADIGM 2: CENTRALIZED / STANDARD FEDAVG (ENCODER + HEAD SHARED)
    # =========================================================================
    print("\n--- Training Paradigm 2: Standard FedAvg (Encoder & Head Shared) ---")
    fedavg_models = [PersonalizedClientModel() for _ in range(num_clients)]
    fedavg_opts = [optim.Adam(fedavg_models[i].parameters(), lr=lr) for i in range(num_clients)]
    
    for r in range(rounds):
        for i, client in enumerate(clients_data):
            for _ in range(epochs_per_round):
                train_local_epoch(fedavg_models[i], client["train_loader"], fedavg_opts[i], client["criterion"])
                
        # Aggregate encoder & head weights
        avg_enc_state = average_state_dicts([model.encoder.state_dict() for model in fedavg_models])
        avg_head_state = average_state_dicts([model.head.state_dict() for model in fedavg_models])
        
        for model in fedavg_models:
            model.encoder.load_state_dict(avg_enc_state)
            model.head.load_state_dict(avg_head_state)
            
    for i, client in enumerate(clients_data):
        metrics = evaluate_model(fedavg_models[i], client["test_loader"])
        for m in metrics.keys():
            results["Centralized/FedAvg"][m].append(metrics[m])
            
    # =========================================================================
    # PARADIGM 3: PERSONALIZED FEDERATED LEARNING (FEDPER - ENCODER SHARED)
    # =========================================================================
    print("\n--- Training Paradigm 3: Personalized Federated Learning (FedPer) ---")
    pfl_models = [PersonalizedClientModel() for _ in range(num_clients)]
    pfl_opts = [optim.Adam(pfl_models[i].parameters(), lr=lr) for i in range(num_clients)]
    
    for r in range(rounds):
        for i, client in enumerate(clients_data):
            for _ in range(epochs_per_round):
                train_local_epoch(pfl_models[i], client["train_loader"], pfl_opts[i], client["criterion"])
                
        # Aggregate shared encoder only
        avg_enc_state = average_state_dicts([model.encoder.state_dict() for model in pfl_models])
        
        for model in pfl_models:
            model.encoder.load_state_dict(avg_enc_state)
            
    for i, client in enumerate(clients_data):
        metrics = evaluate_model(pfl_models[i], client["test_loader"])
        for m in metrics.keys():
            results["Personalized FL"][m].append(metrics[m])

    # Deployable population prior for Aestra soft-read (optional personal estimate).
    # FedPer keeps heads local during training; for product inference we average
    # proj + encoder + head across clients as a research prior (not clinical).
    try:
        from cyclebench.model.sequence_research import save_deployable_checkpoint
        deploy = PersonalizedClientModel()
        deploy.proj.load_state_dict(average_state_dicts([m.proj.state_dict() for m in pfl_models]))
        deploy.encoder.load_state_dict(average_state_dicts([m.encoder.state_dict() for m in pfl_models]))
        deploy.head.load_state_dict(average_state_dicts([m.head.state_dict() for m in pfl_models]))
        all_X = np.concatenate([c["X_train"].reshape(-1, 10) for c in clients_data], axis=0)
        feat_mean = np.mean(all_X, axis=0)
        feat_std = np.std(all_X, axis=0)
        ckpt = save_deployable_checkpoint(
            deploy.state_dict(),
            feat_mean,
            feat_std,
            meta={"source": "multi_symptom_fedper", "rounds": rounds, "n_clients": num_clients},
        )
        print(f"Deployable sequence research checkpoint saved to: {ckpt}")
    except Exception as e:
        print(f"(checkpoint export skipped: {e})")
            
    # =========================================================================
    # PRINT RESULTS SUMMARY
    # =========================================================================
    print("\n" + "="*95)
    print("                      MULTI-SYMPTOM TEMPORAL SIMULATION RESULTS")
    print(f"                      (AVERAGED ACROSS ALL {num_clients} PATIENT CLIENTS)")
    print("="*95)
    print(f"{'Metric':<25} | {'Local Only':<15} | {'Centralized / FedAvg':<22} | {'Personalized FL (pFL)':<22}")
    print("-"*95)
    
    for metric in ["accuracy", "sensitivity", "specificity", "f1"]:
        v_local = np.nanmean(results["Local Only"][metric])
        v_fedavg = np.nanmean(results["Centralized/FedAvg"][metric])
        v_pfl = np.nanmean(results["Personalized FL"][metric])
        print(f"{metric.capitalize():<25} | {v_local:<15.4f} | {v_fedavg:<22.4f} | {v_pfl:<22.4f}")
    print("="*95)
    
    # =========================================================================
    # PLOT COMPARISON GRAPH
    # =========================================================================
    metrics_list = ["accuracy", "sensitivity", "specificity", "f1"]
    metric_labels = ["Accuracy", "Sensitivity", "Specificity", "F1 Score"]
    
    local_means = [np.nanmean(results["Local Only"][m]) for m in metrics_list]
    fedavg_means = [np.nanmean(results["Centralized/FedAvg"][m]) for m in metrics_list]
    pfl_means = [np.nanmean(results["Personalized FL"][m]) for m in metrics_list]
    
    plt.figure(figsize=(12, 7))
    x = np.arange(len(metrics_list))
    width = 0.25
    
    plt.bar(x - width, local_means, width, label='Local Only', color='indianred')
    plt.bar(x, fedavg_means, width, label='Centralized/FedAvg', color='dodgerblue')
    plt.bar(x + width, pfl_means, width, label='Personalized FL (pFL)', color='seagreen')
    
    plt.ylabel('Average Score')
    plt.title(f'Multi-Symptom Temporal Federated Learning Performance (Averaged across {num_clients} Clients)')
    plt.xticks(x, metric_labels)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"\nMulti-symptom temporal simulation plot saved to: {PLOT_PATH}")

    summary = {
        "experiment": "multi_symptom_temporal",
        "task": "binary_cramps_from_hormones_and_symptoms",
        "window": 5,
        "rounds": rounds,
        "n_clients": num_clients,
        "metrics": {
            paradigm: {
                m: float(np.nanmean(results[paradigm][m]))
                for m in ["accuracy", "sensitivity", "specificity", "f1"]
            }
            for paradigm in eval_paradigms
        },
        "honesty_note": (
            "Research simulation of FedPer on mcPHASES patient clients. "
            "Not a clinical diagnostic model. Walkthrough full-run used 30 rounds; "
            "smoke runs may use fewer via PFL_ROUNDS."
        ),
    }
    out_json = RESULTS_DIR / "pfl_multi_symptom.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Metrics written to {out_json}")
    return summary

if __name__ == "__main__":
    main()
