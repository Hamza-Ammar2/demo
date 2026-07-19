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

from federated_learning.paths import FL_DIR, RESULTS_DIR, require_mcphases, rounds as pfl_rounds

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

MCP_HORM_PATH = str(require_mcphases())
OUTPUT_DIR = str(FL_DIR)
PLOT_PATH = str(FL_DIR / "patient_centric_comparison.png")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hormone columns
HORM_COLS = ['lh', 'estrogen', 'pdg']

# 1. Models Definition (Wearable Scale)

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim=3, output_dim=8):
        super(ProjectionLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.LayerNorm(8),
            nn.ReLU(),
            nn.Linear(8, output_dim)
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
            nn.Linear(8, 1)  # Output raw logits for BCEWithLogitsLoss
        )
    def forward(self, x):
        return self.net(x)

class PersonalizedClientModel(nn.Module):
    def __init__(self, input_dim=3, latent_dim=8):
        super(PersonalizedClientModel, self).__init__()
        self.proj = ProjectionLayer(input_dim, latent_dim)
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
    
    # Map cramps symptoms: Moderate, High, Very High to 1 (cramps presence), else 0
    df['cramps_binary'] = df['cramps'].apply(lambda val: 1 if val in ['Moderate', 'High', 'Very High'] else 0)
    
    # Clean rows: keep where at least one hormone is present
    df = df.dropna(subset=HORM_COLS, how='all')
    
    # Process each subject as a patient client
    clients_data = []
    unique_subject_ids = df['id'].unique()
    
    for sub_id in unique_subject_ids:
        sub_df = df[df['id'] == sub_id].sort_values(by='day_in_study').copy()
        
        # Log-transform
        for col in HORM_COLS:
            sub_df[col] = np.log1p(sub_df[col])
            
        # Interpolate and pad longitudinal sequence locally for this patient
        sub_df[HORM_COLS] = sub_df[HORM_COLS].interpolate(method='linear').ffill().bfill()
        sub_df = sub_df.dropna(subset=HORM_COLS)
        
        # Minimum sequence length check
        if len(sub_df) < 30:
            continue
            
        X = sub_df[HORM_COLS].values
        y = sub_df['cramps_binary'].values
        
        # Randomized split (80% train, 20% test) to handle cycle-dependent symptom sparsity
        n_samples = len(X)
        shuffled_idx = np.random.permutation(n_samples)
        split = int(0.8 * n_samples)
        
        train_idx = shuffled_idx[:split]
        test_idx = shuffled_idx[split:]
        
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        
        # Ensure we don't have degenerate local datasets with only one class label in train or test
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
            
        # Standardize locally to the patient's baseline distribution
        mean = np.mean(X_tr, axis=0)
        std = np.std(X_tr, axis=0)
        std[std == 0] = 1.0
        
        X_tr_scaled = (X_tr - mean) / std
        X_te_scaled = (X_te - mean) / std
        
        # Compute pos_weight for BCEWithLogitsLoss based on training distribution imbalance
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
    print(f"\nFiltered McPhases Cohort: Selected {num_clients} unique patient clients (with >= 30 valid days and balanced train/test symptom labels).")
    if num_clients == 0:
        print("Error: No patients matched the selection criteria. Exiting.")
        return
        
    # Helper to construct PyTorch loader
    def build_loader(X, y, batch_size=8, shuffle=True):
        tensor_x = torch.tensor(X, dtype=torch.float32)
        tensor_y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(tensor_x, tensor_y)
        # Avoid BatchNorm1d crash by dropping last single-element batch during training
        drop_last = shuffle and (len(dataset) > batch_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        
    # Initialize Datasets and Loaders for all clients
    for client in clients_data:
        client["train_loader"] = build_loader(client["X_train"], client["y_train"])
        client["test_loader"] = build_loader(client["X_test"], client["y_test"], shuffle=False)
        
    # Setup training hyperparameters
    lr = 0.003
    epochs_per_round = 3
    rounds = pfl_rounds(30)
    print(f"Running patient-centric pFL with rounds={rounds}")
    
    # To accumulate evaluation metrics
    # Paradigms: Local Only, Centralized/FedAvg, Personalized FL
    eval_paradigms = ["Local Only", "Centralized/FedAvg", "Personalized FL"]
    results = {p: {metric: [] for metric in ["accuracy", "sensitivity", "specificity", "f1"]} for p in eval_paradigms}
    
    # =========================================================================
    # PARADIGM 1: LOCAL ONLY (TRAINED COMPLETELY INDEPENDENTLY)
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
                
        # Aggregate encoder & head weights globally
        avg_enc_state = average_state_dicts([model.encoder.state_dict() for model in fedavg_models])
        avg_head_state = average_state_dicts([model.head.state_dict() for model in fedavg_models])
        
        # Load weights back to keep models synchronized
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
                
        # Aggregate ONLY the representation encoder weights
        avg_enc_state = average_state_dicts([model.encoder.state_dict() for model in pfl_models])
        
        # Load back to maintain federated base
        for model in pfl_models:
            model.encoder.load_state_dict(avg_enc_state)
            
    for i, client in enumerate(clients_data):
        metrics = evaluate_model(pfl_models[i], client["test_loader"])
        for m in metrics.keys():
            results["Personalized FL"][m].append(metrics[m])
            
    # =========================================================================
    # PRINT SUMMARY AND AGGREGATE RESULTS
    # =========================================================================
    print("\n" + "="*95)
    print("                      PATIENT-CENTRIC WEARABLE SIMULATION RESULTS")
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
    plt.title(f'Patient-Centric Federated Learning Performance (Averaged across {num_clients} Wearable Patients)')
    plt.xticks(x, metric_labels)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(PLOT_PATH)
    plt.close()
    print(f"\nPatient-centric simulation comparison plot saved to: {PLOT_PATH}")

if __name__ == "__main__":
    main()
