"""Personalized FedPer-style GRU for cycle-phase prediction.

Upgrades from Hamza ``fed`` @ 6ddf762 (database setup):
  - Anonymous client id + explicit research consent
  - Optional Hugging Face dataset sync (consent-gated)
  - Shared ``global_scaler.pt`` for train/infer/sync
  - Peer logs from HF when configured

Still on this merge branch:
  - Offline fallback: local ``results/hf_export/data`` or mcPHASES
  - No fabricated sync accuracies
  - Pad×5 honesty flags; lazy torch import
  - Aestra UI wiring lives outside this module
"""

from __future__ import annotations

import copy
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LOCAL_DATA_PATH = RESULTS_DIR / "local_patient_data.csv"
LOCAL_MODEL_PATH = MODELS_DIR / "local_patient_model.pt"
GLOBAL_MODEL_PATH = MODELS_DIR / "global_pfl_model.pt"
GLOBAL_SCALER_PATH = MODELS_DIR / "global_scaler.pt"
GLOBAL_META_PATH = MODELS_DIR / "global_pfl_model.meta.npz"
HF_EXPORT_DATA = RESULTS_DIR / "hf_export" / "data"
CLIENT_ID_PATH = RESULTS_DIR / "client_id.txt"
CONSENT_PATH = RESULTS_DIR / "user_consent.txt"

CLASSES = ["Menstrual", "Follicular", "Fertility", "Luteal"]
FEATURE_COLS = [
    "headaches_ord", "cramps_ord", "sorebreasts_ord", "fatigue_ord", "sleepissue_ord",
    "moodswing_ord", "stress_ord", "foodcravings_ord", "indigestion_ord", "bloating_ord",
    "appetite_ord", "sleep_minutes", "sleep_awake", "sleep_efficiency", "resting_hr",
    "steps_sum", "stress_score_mean", "wrist_temp_delta", "glucose_mean", "hrv_rmssd",
    "is_weekend",
]
WINDOW = 5


def _torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for pFL. Install with: make install-pfl"
        ) from e
    return torch, nn, optim, DataLoader, TensorDataset


def average_state_dicts(state_dicts: list[dict]) -> dict:
    if not state_dicts:
        raise ValueError("average_state_dicts requires at least one state_dict")
    avg = copy.deepcopy(state_dicts[0])
    for key in avg:
        for sd in state_dicts[1:]:
            avg[key] = avg[key] + sd[key]
        avg[key] = avg[key] / float(len(state_dicts))
    return avg


def get_anonymous_client_id() -> str:
    if CLIENT_ID_PATH.exists():
        return CLIENT_ID_PATH.read_text().strip()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    client_id = f"client_{uuid.uuid4().hex[:12]}"
    CLIENT_ID_PATH.write_text(client_id)
    return client_id


def check_user_consent() -> bool:
    if not CONSENT_PATH.exists():
        return False
    return CONSENT_PATH.read_text().strip() in ("1", "true", "True")


def set_user_consent(consent: bool) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CONSENT_PATH.write_text("1" if consent else "0")


def _hf_creds() -> tuple[str | None, str | None]:
    return os.environ.get("HF_TOKEN"), os.environ.get("HF_DATASET_REPO")


def sync_local_to_huggingface() -> dict[str, Any]:
    """Upload local logs to HF dataset repo (requires consent + credentials)."""
    if not check_user_consent():
        return {"ok": False, "error": "Sync aborted: user consent required."}
    hf_token, hf_repo = _hf_creds()
    if not hf_token or not hf_repo:
        return {"ok": False, "error": "Hugging Face credentials not configured (HF_TOKEN / HF_DATASET_REPO)."}
    if not LOCAL_DATA_PATH.exists():
        return {"ok": True, "message": "No local logs to upload."}
    client_id = get_anonymous_client_id()
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True, token=hf_token)
        remote_path = f"data/{client_id}.csv"
        api.upload_file(
            path_or_fileobj=str(LOCAL_DATA_PATH),
            path_in_repo=remote_path,
            repo_id=hf_repo,
            repo_type="dataset",
            token=hf_token,
        )
        return {"ok": True, "client_id": client_id, "uploaded_file": remote_path}
    except Exception as e:
        return {"ok": False, "error": f"Failed to upload to Hugging Face: {e}"}


def _build_modules():
    torch, nn, *_ = _torch()

    class GRUProjection(nn.Module):
        def __init__(self, input_dim=21, hidden_dim=16, output_dim=8):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)
            self.ln = nn.LayerNorm(output_dim)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.ln(self.fc(out[:, -1, :]))

    class SharedEncoder(nn.Module):
        def __init__(self, dim=8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Linear(16, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)

    class DecisionHead(nn.Module):
        def __init__(self, dim=8, num_classes=4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, 8),
                nn.ReLU(),
                nn.Linear(8, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    class PersonalizedClientModel(nn.Module):
        def __init__(self, input_dim=21, hidden_dim=16, latent_dim=8, num_classes=4):
            super().__init__()
            self.proj = GRUProjection(input_dim, hidden_dim, latent_dim)
            self.encoder = SharedEncoder(latent_dim)
            self.head = DecisionHead(latent_dim, num_classes)

        def forward(self, x):
            return self.head(self.encoder(self.proj(x)))

    return PersonalizedClientModel


def _load_state(model, path: Path) -> None:
    torch, *_ = _torch()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict) and "state_dict" in blob:
        model.load_state_dict(blob["state_dict"])
    else:
        model.load_state_dict(blob)


def _save_scaler(mean: np.ndarray, std: np.ndarray) -> None:
    torch, *_ = _torch()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    torch.save({"mean": mean, "std": std}, GLOBAL_SCALER_PATH)
    np.savez(GLOBAL_META_PATH, mean=mean, std=std)


def _save_global(model, mean: np.ndarray, std: np.ndarray) -> None:
    torch, *_ = _torch()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, GLOBAL_MODEL_PATH)
    _save_scaler(mean, std)


def load_global_scaler() -> tuple[np.ndarray, np.ndarray]:
    """Prefer Hamza's global_scaler.pt, then meta.npz, else zeros/ones."""
    torch, *_ = _torch()
    if GLOBAL_SCALER_PATH.exists():
        try:
            scaler = torch.load(GLOBAL_SCALER_PATH, map_location="cpu", weights_only=False)
            return np.asarray(scaler["mean"], dtype=np.float32), np.asarray(scaler["std"], dtype=np.float32)
        except Exception:
            pass
    if GLOBAL_META_PATH.exists():
        z = np.load(GLOBAL_META_PATH)
        return z["mean"].astype(np.float32), z["std"].astype(np.float32)
    return np.zeros(len(FEATURE_COLS), dtype=np.float32), np.ones(len(FEATURE_COLS), dtype=np.float32)


def ensure_global_norm() -> tuple[np.ndarray, np.ndarray]:
    mean, std = load_global_scaler()
    if GLOBAL_SCALER_PATH.exists() or GLOBAL_META_PATH.exists():
        return mean, std
    # Build from available peer tables once
    chunks = []
    for _label, df in _iter_peer_frames(max_peers=50, exclude_own=False):
        X_o, _ = build_sequential_dataset(df, w=WINDOW)
        if X_o is None:
            continue
        chunks.append(X_o.reshape(-1, len(FEATURE_COLS)))
    if not chunks:
        return mean, std
    flat = np.concatenate(chunks, axis=0)
    mean = np.mean(flat, axis=0).astype(np.float32)
    std = np.std(flat, axis=0).astype(np.float32)
    std[std == 0] = 1.0
    _save_scaler(mean, std)
    return mean, std


def save_local_log(features: dict, phase: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    row = {f: float(features.get(f, 0.0) or 0.0) for f in FEATURE_COLS}
    row["phase"] = phase
    df = pd.DataFrame([row])
    if LOCAL_DATA_PATH.exists():
        df.to_csv(LOCAL_DATA_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOCAL_DATA_PATH, mode="w", header=True, index=False)

    if check_user_consent() and _hf_creds()[0] and _hf_creds()[1]:
        threading.Thread(target=sync_local_to_huggingface, daemon=True).start()


def _ensure_feature_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0.0
    if "phase" not in out.columns:
        out["phase"] = CLASSES[0]
    return out


def build_sequential_dataset(df: pd.DataFrame, w: int = WINDOW):
    df = _ensure_feature_cols(df)
    df[FEATURE_COLS] = (
        df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
        .interpolate(method="linear").ffill().bfill().fillna(0.0)
    )
    if len(df) < w:
        return None, None

    X_vals = df[FEATURE_COLS].values.astype(np.float32)
    phase_to_idx = {name: i for i, name in enumerate(CLASSES)}
    y_vals = df["phase"].map(phase_to_idx).fillna(0).values.astype(int)

    X_seq, y_seq = [], []
    for t in range(w - 1, len(X_vals)):
        X_seq.append(X_vals[t - w + 1 : t + 1])
        y_seq.append(y_vals[t])
    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.int64)


def _iter_peer_frames(max_peers: int = 11, exclude_own: bool = True) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield (source_label, dataframe) from HF → local export → mcPHASES."""
    own = f"data/{get_anonymous_client_id()}.csv" if exclude_own else None
    yielded = 0

    hf_token, hf_repo = _hf_creds()
    if hf_token and hf_repo:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files

            files = list_repo_files(repo_id=hf_repo, repo_type="dataset", token=hf_token)
            csv_files = [f for f in files if f.startswith("data/") and f.endswith(".csv")]
            if own:
                csv_files = [f for f in csv_files if f != own]
            for f in csv_files:
                if yielded >= max_peers:
                    return
                try:
                    path = hf_hub_download(
                        repo_id=hf_repo, filename=f, repo_type="dataset", token=hf_token
                    )
                    yield f"hf:{f}", pd.read_csv(path)
                    yielded += 1
                except Exception:
                    continue
            if yielded:
                return
        except Exception:
            pass

    if HF_EXPORT_DATA.exists():
        own_name = Path(own).name if own else None
        for path in sorted(HF_EXPORT_DATA.glob("client_*.csv")):
            if yielded >= max_peers:
                return
            if own_name and path.name == own_name:
                continue
            try:
                yield f"export:{path.name}", pd.read_csv(path)
                yielded += 1
            except Exception:
                continue
        if yielded:
            return

    try:
        from cyclebench.model.features_mcphases import build_mcphases_table

        mcp_df = build_mcphases_table()
        for pid in list(mcp_df["id"].unique())[:max_peers]:
            if yielded >= max_peers:
                return
            sub = mcp_df[mcp_df["id"] == pid].copy()
            yield f"mcphases:{pid}", sub
            yielded += 1
    except Exception:
        return


def train_global_pfl_model(epochs: int = 15, seed: int = 42) -> dict[str, Any]:
    torch, nn, optim, DataLoader, TensorDataset = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_all, y_all = [], []
    sources = []
    for label, sub in _iter_peer_frames(max_peers=50, exclude_own=True):
        X_o, y_o = build_sequential_dataset(sub, w=WINDOW)
        if X_o is None or len(X_o) < 5:
            continue
        X_all.append(X_o)
        y_all.append(y_o)
        sources.append(label)

    if not X_all:
        return {
            "ok": False,
            "error": (
                "No sequential peer logs. Configure HF_TOKEN/HF_DATASET_REPO, run "
                "scripts/prepare_hf_dataset.py, or place mcPHASES under data/mcphases/."
            ),
        }

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    mean = np.mean(X.reshape(-1, len(FEATURE_COLS)), axis=0)
    std = np.std(X.reshape(-1, len(FEATURE_COLS)), axis=0)
    std[std == 0] = 1.0
    X_scaled = ((X.reshape(-1, len(FEATURE_COLS)) - mean) / std).reshape(X.shape)

    Model = _build_modules()
    model = Model()
    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        ),
        batch_size=32,
        shuffle=True,
    )
    opt = optim.Adam(model.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()

    _save_global(model, mean, std)
    backend = "hf" if any(s.startswith("hf:") for s in sources) else (
        "hf_export" if any(s.startswith("export:") for s in sources) else "mcphases"
    )
    return {
        "ok": True,
        "global_model_path": str(GLOBAL_MODEL_PATH),
        "scaler_path": str(GLOBAL_SCALER_PATH),
        "samples": int(len(X)),
        "peers_loaded": len(sources),
        "peer_backend": backend,
        "window": WINDOW,
        "task": "hormonal_state_phase",
    }


def train_local_pfl(epochs: int = 15) -> dict[str, Any]:
    torch, nn, optim, DataLoader, TensorDataset = _torch()
    if not LOCAL_DATA_PATH.exists():
        return {"ok": False, "error": "No logged data. Call /models/hormonal-state (or chip intake) first."}

    df = pd.read_csv(LOCAL_DATA_PATH)
    if len(df) < 8:
        return {
            "ok": False,
            "error": f"Need ≥8 logged days (have {len(df)}). Pad×5 inference still works with fewer.",
        }

    X, y = build_sequential_dataset(df, w=WINDOW)
    if X is None or len(X) < 3:
        return {"ok": False, "error": "Not enough rows to build sliding windows (need ≥5 days)."}

    mean, std = load_global_scaler()
    X_scaled = ((X.reshape(-1, len(FEATURE_COLS)) - mean) / std).reshape(X.shape)

    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    n = len(dataset)
    split = max(1, int(0.8 * n))
    train_ds = torch.utils.data.Subset(dataset, range(split))
    test_ds = torch.utils.data.Subset(dataset, range(split, n))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=(len(train_ds) > 4))

    Model = _build_modules()
    model = Model()
    if LOCAL_MODEL_PATH.exists():
        _load_state(model, LOCAL_MODEL_PATH)
    elif GLOBAL_MODEL_PATH.exists():
        _load_state(model, GLOBAL_MODEL_PATH)
    else:
        g = train_global_pfl_model()
        if not g.get("ok"):
            return g
        _load_state(model, GLOBAL_MODEL_PATH)

    opt = optim.Adam(model.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    for _ in range(epochs):
        epoch_loss = 0.0
        for bx, by in train_loader:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * bx.size(0)
        total_loss += epoch_loss / max(len(train_ds), 1)

    torch.save({"state_dict": model.state_dict()}, LOCAL_MODEL_PATH)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(len(test_ds)):
            bx, by = test_ds[i]
            pred = int(torch.argmax(model(bx.unsqueeze(0)), dim=1).item())
            correct += int(pred == int(by.item()))
            total += 1
    test_acc = (correct / total) if total else None

    return {
        "ok": True,
        "train_loss": round(total_loss / epochs, 4),
        "test_accuracy": round(test_acc, 4) if test_acc is not None else None,
        "test_n": total,
        "total_days_logged": len(df),
        "sequences_trained": len(train_ds),
        "honesty_note": (
            "Local labels are whatever phase was stored in the CSV (often the model's own "
            "prediction — not ground-truth phases)."
        ),
    }


def federated_sync_pfl(peer_epochs: int = 2, max_peers: int = 11) -> dict[str, Any]:
    """One FedPer round: average peer encoders (+ local). Peers from HF / export / mcPHASES."""
    torch, nn, optim, DataLoader, TensorDataset = _torch()

    if check_user_consent():
        sync_local_to_huggingface()

    if not LOCAL_MODEL_PATH.exists():
        if not GLOBAL_MODEL_PATH.exists():
            g = train_global_pfl_model()
            if not g.get("ok"):
                return g
        Model = _build_modules()
        m = Model()
        _load_state(m, GLOBAL_MODEL_PATH)
        torch.save({"state_dict": m.state_dict()}, LOCAL_MODEL_PATH)

    Model = _build_modules()
    local_model = Model()
    _load_state(local_model, LOCAL_MODEL_PATH)
    mean_g, std_g = load_global_scaler()

    test_ds_l = None
    acc_before = None
    if LOCAL_DATA_PATH.exists():
        df_local = pd.read_csv(LOCAL_DATA_PATH)
        X_local, y_local = build_sequential_dataset(df_local, w=WINDOW)
        if X_local is not None and len(X_local) >= 2:
            Xs = ((X_local.reshape(-1, len(FEATURE_COLS)) - mean_g) / std_g).reshape(X_local.shape)
            dataset_l = TensorDataset(
                torch.tensor(Xs, dtype=torch.float32),
                torch.tensor(y_local, dtype=torch.long),
            )
            split_idx = max(1, int(0.8 * len(dataset_l)))
            test_ds_l = torch.utils.data.Subset(dataset_l, range(split_idx, len(dataset_l)))
            if len(test_ds_l) > 0:
                local_model.eval()
                correct = 0
                with torch.no_grad():
                    for i in range(len(test_ds_l)):
                        bx, by = test_ds_l[i]
                        pred = int(torch.argmax(local_model(bx.unsqueeze(0)), dim=1).item())
                        correct += int(pred == int(by.item()))
                acc_before = correct / len(test_ds_l)

    other_encoders = []
    peer_backend = "none"
    for label, sub in _iter_peer_frames(max_peers=max_peers, exclude_own=True):
        X_o, y_o = build_sequential_dataset(sub, w=WINDOW)
        if X_o is None or len(X_o) < 5:
            continue
        Xs = ((X_o.reshape(-1, len(FEATURE_COLS)) - mean_g) / std_g).reshape(X_o.shape)
        loader_o = DataLoader(
            TensorDataset(
                torch.tensor(Xs, dtype=torch.float32),
                torch.tensor(y_o, dtype=torch.long),
            ),
            batch_size=8,
            shuffle=True,
            drop_last=(len(X_o) > 8),
        )
        o_model = Model()
        if GLOBAL_MODEL_PATH.exists():
            _load_state(o_model, GLOBAL_MODEL_PATH)
        o_opt = optim.Adam(o_model.parameters(), lr=0.005)
        o_crit = nn.CrossEntropyLoss()
        o_model.train()
        for _ in range(peer_epochs):
            for bx, by in loader_o:
                o_opt.zero_grad()
                loss = o_crit(o_model(bx), by)
                loss.backward()
                o_opt.step()
        other_encoders.append(o_model.encoder.state_dict())
        peer_backend = label.split(":", 1)[0]

    if not other_encoders:
        return {"ok": False, "error": "No peer clients with enough sequential data."}

    encoders = [local_model.encoder.state_dict()] + other_encoders
    local_model.encoder.load_state_dict(average_state_dicts(encoders))
    torch.save({"state_dict": local_model.state_dict()}, LOCAL_MODEL_PATH)

    acc_after = None
    if test_ds_l is not None and len(test_ds_l) > 0:
        local_model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(len(test_ds_l)):
                bx, by = test_ds_l[i]
                pred = int(torch.argmax(local_model(bx.unsqueeze(0)), dim=1).item())
                correct += int(pred == int(by.item()))
        acc_after = correct / len(test_ds_l)

    out: dict[str, Any] = {
        "ok": True,
        "peers_synced": len(other_encoders),
        "peer_backend": peer_backend,
        "simulation": peer_backend != "hf",
        "honesty_note": (
            "FedPer encoder average over peer logs. "
            "HF when credentials are set; else local export/mcPHASES. "
            "Accuracies only when a local eval split exists (no fabricated numbers)."
        ),
    }
    if acc_before is not None:
        out["accuracy_before_sync"] = round(acc_before, 4)
    if acc_after is not None:
        out["accuracy_after_sync"] = round(acc_after, 4)
        if acc_before is not None:
            out["performance_gain"] = round(acc_after - acc_before, 4)
    return out


def run_pfl_inference(features: dict) -> dict[str, Any]:
    torch, *_ = _torch()
    Model = _build_modules()
    model = Model()

    if LOCAL_MODEL_PATH.exists():
        _load_state(model, LOCAL_MODEL_PATH)
    elif GLOBAL_MODEL_PATH.exists():
        _load_state(model, GLOBAL_MODEL_PATH)
    else:
        g = train_global_pfl_model()
        if not g.get("ok"):
            raise FileNotFoundError(g.get("error", "PFL checkpoint missing"))
        _load_state(model, GLOBAL_MODEL_PATH)

    model.eval()

    if LOCAL_DATA_PATH.exists():
        df = pd.read_csv(LOCAL_DATA_PATH)
    else:
        df = pd.DataFrame(columns=FEATURE_COLS)

    curr = np.array([float(features.get(f, 0.0) or 0.0) for f in FEATURE_COLS], dtype=np.float32)
    padded = False
    if len(df) >= 4 and all(c in df.columns for c in FEATURE_COLS):
        last = df.tail(4)[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        seq = np.vstack([last, curr.reshape(1, -1)])
    else:
        seq = np.repeat(curr.reshape(1, -1), WINDOW, axis=0)
        padded = True

    mean, std = ensure_global_norm()
    seq_scaled = (seq - mean) / std

    with torch.no_grad():
        logits = model(torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0))
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

    pred_idx = int(np.argmax(probs))
    return {
        "task": "hormonal_state_phase",
        "predicted_state": CLASSES[pred_idx],
        "probabilities": {name: round(float(p), 4) for name, p in zip(CLASSES, probs)},
        "confidence": round(float(probs[pred_idx]), 4),
        "model": "Personalized_GRU_FedPer",
        "window": WINDOW,
        "sequence_padded": padded,
        "explanation": [
            {"feature": f, "value": round(float(features.get(f, 0.0) or 0.0), 2)}
            for f in FEATURE_COLS[:5]
        ],
        "honesty_note": (
            "Pad×5 used because <4 prior days were logged."
            if padded
            else "Sequence built from recent local logs + current day."
        ),
    }
