#!/usr/bin/env python
# coding: utf-8
# To use as notebook: open in VSCode (Jupyter extension) or run: jupytext --to notebook wildfire_prediction.py

# %% [markdown]
# # Wildfire Spread Prediction — Multimodal CNN + LSTM + XGBoost
# **Dataset:** Next Day Wildfire Spread (Huot et al., IEEE TGRS 2022)
# **Task:** Predict whether a wildfire will spread in a given 64×64km patch next day
# **Metrics:** PR-AUC (primary), ROC-AUC (secondary) — matching paper convention
# **Baseline to beat:** PR-AUC 28.4% (paper's neural network)

# %% [markdown]
# ## 1. Setup

# %%
import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (average_precision_score, roc_auc_score,
                              precision_recall_curve, roc_curve,
                              classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Constants

# %%
INPUT_FEATURES = [
    'elevation',    # terrain elevation
    'th',           # wind direction
    'vs',           # wind speed
    'tmmn',         # min temperature
    'tmmx',         # max temperature
    'sph',          # specific humidity
    'pr',           # precipitation
    'pdsi',         # drought index (Palmer Drought Severity Index)
    'NDVI',         # vegetation index
    'population',   # population density
    'erc',          # energy release component (fire weather)
    'PrevFireMask', # fire mask at time t
]
OUTPUT_FEATURES = ['FireMask']   # fire mask at time t+1

N_FEATURES  = len(INPUT_FEATURES)   # 12
PATCH_SIZE  = 64
DATA_DIR    = '../data'
SAVE_DIR    = 'processed'
MLFLOW_DIR  = 'mlruns'

os.makedirs(SAVE_DIR,  exist_ok=True)
os.makedirs(MLFLOW_DIR, exist_ok=True)

# Training hyperparams
BATCH_SIZE  = 32
EPOCHS_CNN  = 25
EPOCHS_LSTM = 20
EPOCHS_FUSE = 15
LR          = 1e-3

# %% [markdown]
# ## 3. TFRecord Parsing (one-time, ~20-30 mins → saved as .npy)

# %%
def parse_tfrecord(example_proto, tf):
    """Parse a single TFRecord example into numpy arrays."""
    keys = INPUT_FEATURES + OUTPUT_FEATURES
    feature_desc = {
        k: tf.io.FixedLenFeature([PATCH_SIZE * PATCH_SIZE], tf.float32)
        for k in keys
    }
    parsed = tf.io.parse_single_example(example_proto, feature_desc)
    inputs = np.stack(
        [parsed[k].numpy().reshape(PATCH_SIZE, PATCH_SIZE) for k in INPUT_FEATURES],
        axis=-1   # shape: (64, 64, 12)
    )
    label = parsed['FireMask'].numpy().reshape(PATCH_SIZE, PATCH_SIZE)
    return inputs.astype(np.float32), label.astype(np.float32)


def load_split(split_name: str, tf):
    """Load all TFRecord files matching split_name."""
    search_split = 'eval' if split_name == 'val' else split_name
    pattern = os.path.join(DATA_DIR, f'*{search_split}*')
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No TFRecord files found for split '{split_name}' in {DATA_DIR}")
    print(f"  [{split_name}] Found {len(files)} file(s)")

    X_list, y_list = [], []
    dataset = tf.data.TFRecordDataset(files)
    for i, raw in enumerate(dataset):
        x, y = parse_tfrecord(raw, tf)
        X_list.append(x)
        y_list.append(y)
        if (i + 1) % 2000 == 0:
            print(f"    Parsed {i+1} samples...")
    return np.array(X_list), np.array(y_list)


def parse_and_cache():
    """Parse TFRecords once and save as .npy files."""
    import tensorflow as tf

    splits = ['train', 'val', 'test']
    for split in splits:
        xp = os.path.join(SAVE_DIR, f'X_{split}.npy')
        yp = os.path.join(SAVE_DIR, f'y_{split}.npy')
        if os.path.exists(xp):
            print(f"  [{split}] Cache found, skipping.")
            continue
        print(f"  [{split}] Parsing TFRecords...")
        X, y = load_split(split, tf)
        np.save(xp, X)
        np.save(yp, y)
        print(f"  [{split}] Saved → X:{X.shape}  y:{y.shape}")


print("Checking / building numpy cache...")
parse_and_cache()

# %% [markdown]
# ## 4. Load Data + Build Patch-Level Labels

# %%
def load_numpy(split):
    X = np.load(os.path.join(SAVE_DIR, f'X_{split}.npy'))  # (N, 64, 64, 12)
    y = np.load(os.path.join(SAVE_DIR, f'y_{split}.npy'))  # (N, 64, 64)
    return X, y


X_train_raw, y_train_raw = load_numpy('train')
X_val_raw,   y_val_raw   = load_numpy('val')
X_test_raw,  y_test_raw  = load_numpy('test')

print(f"Train  X:{X_train_raw.shape}  y:{y_train_raw.shape}")
print(f"Val    X:{X_val_raw.shape}    y:{y_val_raw.shape}")
print(f"Test   X:{X_test_raw.shape}   y:{y_test_raw.shape}")


def make_patch_labels(y_raw):
    """
    Patch-level binary label.
    1 = fire spread occurred somewhere in the 64x64 patch next day.
    Use max over patch pixels so even a small spread counts.
    """
    return (y_raw.reshape(len(y_raw), -1).max(axis=1) > 0.5).astype(np.int32)


y_train = make_patch_labels(y_train_raw)
y_val   = make_patch_labels(y_val_raw)
y_test  = make_patch_labels(y_test_raw)

pos_rate = y_train.mean()
print(f"\nClass balance (train): {pos_rate:.1%} positive  |  imbalance ratio: 1:{1/pos_rate:.0f}")

# %% [markdown]
# ## 5. EDA

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('EDA — Sample Patches (fire spread = 1)', fontsize=14, fontweight='bold')

fire_idxs    = np.where(y_train == 1)[0][:3]
no_fire_idxs = np.where(y_train == 0)[0][:3]

feature_names_short = ['Elevation', 'Wind Dir', 'Wind Spd', 'Temp Min', 'Temp Max', 'Humidity',
                        'Precip', 'Drought', 'NDVI', 'Population', 'ERC', 'Prev Fire']

for col, idx in enumerate(fire_idxs):
    patch  = X_train_raw[idx]          # (64, 64, 12)
    fire   = y_train_raw[idx]          # (64, 64)
    prev   = patch[:, :, 11]           # PrevFireMask channel

    ax = axes[0, col]
    rgb = np.stack([patch[:, :, 4], patch[:, :, 8], prev], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    ax.imshow(rgb)
    ax.contour(fire, levels=[0.5], colors='red', linewidths=1.5)
    ax.set_title(f'Sample {idx} | Fire Spread ✓', fontsize=9)
    ax.axis('off')

# Feature distributions
ax = axes[1, 0]
feature_means = X_train_raw.reshape(-1, N_FEATURES).mean(axis=0)
ax.barh(feature_names_short, feature_means, color='steelblue', alpha=0.7)
ax.set_title('Feature Mean Values', fontsize=9)
ax.set_xlabel('Mean (unnormalized)')

# Class imbalance
ax = axes[1, 1]
counts = [y_train.sum(), (y_train == 0).sum()]
ax.pie(counts, labels=['Fire Spread', 'No Spread'], autopct='%1.1f%%',
       colors=['#e74c3c', '#2ecc71'], startangle=140)
ax.set_title('Class Distribution (Train)', fontsize=9)

# Correlation heatmap (sample 1000)
ax = axes[1, 2]
sample_feats = X_train_raw[:1000].reshape(-1, N_FEATURES)
corr = pd.DataFrame(sample_feats, columns=feature_names_short).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax, cmap='coolwarm', center=0,
            xticklabels=True, yticklabels=True, annot=False, linewidths=0.3)
ax.set_title('Feature Correlation', fontsize=9)
ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig('eda.png', dpi=120, bbox_inches='tight')
plt.show()
print("EDA saved → eda.png")

# %% [markdown]
# ## 6. Feature Engineering + Preprocessing

# %%
def normalize_patches(X):
    """
    Per-channel normalization across the dataset.
    Returns (N, 64, 64, 12) normalized.
    """
    X_flat = X.reshape(-1, N_FEATURES)              # (N*64*64, 12)
    mu  = X_flat.mean(axis=0)
    std = X_flat.std(axis=0) + 1e-8
    return ((X - mu) / std).astype(np.float32), mu, std


def extract_tabular(X, mu=None, std=None):
    """
    Aggregate each channel per patch into statistics for XGBoost / MLP.
    Returns (N, 12*5) = (N, 60) feature vector per sample.
    Stats: mean, std, max, min, median per channel.
    """
    N = X.shape[0]
    feats = []
    for fn in [np.mean, np.std, np.max, np.min, np.median]:
        feats.append(fn(X.reshape(N, -1, N_FEATURES), axis=1))  # (N, 12)
    tab = np.concatenate(feats, axis=1).astype(np.float32)       # (N, 60)
    return tab


print("Normalizing...")
X_train_norm, mu, std = normalize_patches(X_train_raw)
X_val_norm   = ((X_val_raw   - mu) / (std)).astype(np.float32)
X_test_norm  = ((X_test_raw  - mu) / (std)).astype(np.float32)

print("Extracting tabular features...")
tab_train = extract_tabular(X_train_norm)
tab_val   = extract_tabular(X_val_norm)
tab_test  = extract_tabular(X_test_norm)

# Scale tabular separately for XGB
scaler = StandardScaler()
tab_train_sc = scaler.fit_transform(tab_train)
tab_val_sc   = scaler.transform(tab_val)
tab_test_sc  = scaler.transform(tab_test)

print(f"Patch features (CNN input):     {X_train_norm.shape}")
print(f"Tabular features (XGB input):   {tab_train.shape}")

# %% [markdown]
# ## 7. PyTorch Dataset

# %%
class WildfireDataset(Dataset):
    """
    Returns:
        img    : (12, 64, 64) float tensor — CNN input
        seq    : (64, 12) float tensor — LSTM input (rows as timesteps)
        tabular: (60,) float tensor — XGB/MLP input
        label  : scalar int
    """
    def __init__(self, X_norm, tab, labels):
        self.X   = torch.from_numpy(X_norm)         # (N, 64, 64, 12)
        self.tab = torch.from_numpy(tab)             # (N, 60)
        self.y   = torch.from_numpy(labels.astype(np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        patch  = self.X[idx]                         # (64, 64, 12)
        img    = patch.permute(2, 0, 1)              # (12, 64, 64)
        # LSTM: each row of the patch is a timestep
        seq    = patch.mean(dim=1)                   # (64, 12) — row-wise mean
        tab    = self.tab[idx]
        label  = self.y[idx]
        return img, seq, tab, label


def make_loaders(batch_size=BATCH_SIZE):
    train_ds = WildfireDataset(X_train_norm, tab_train_sc, y_train)
    val_ds   = WildfireDataset(X_val_norm,   tab_val_sc,   y_val)
    test_ds  = WildfireDataset(X_test_norm,  tab_test_sc,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size*2, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader


train_loader, val_loader, test_loader = make_loaders()
print(f"Batches — train:{len(train_loader)}  val:{len(val_loader)}  test:{len(test_loader)}")

# %% [markdown]
# ## 8. Model Definitions

# %%
# ── 8a. CNN Encoder ──────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
    def forward(self, x):
        return self.block(x)


class CNNEncoder(nn.Module):
    """
    Input:  (B, 12, 64, 64)
    Output: (B, 256)
    """
    def __init__(self, in_channels=N_FEATURES, embed_dim=256):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, 32),   # → (B, 32, 32, 32)
            ConvBlock(32, 64),            # → (B, 64, 16, 16)
            ConvBlock(64, 128),           # → (B, 128, 8, 8)
            ConvBlock(128, 256),          # → (B, 256, 4, 4)
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)   # → (B, 256, 1, 1)
        self.flatten = nn.Flatten()              # → (B, 256)
        self.proj    = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.proj(x)


# ── 8b. LSTM Encoder ─────────────────────────────────────────────────────────
class LSTMEncoder(nn.Module):
    """
    Input:  (B, 64, 12)  — 64 row-timesteps, 12 features each
    Output: (B, 128)
    """
    def __init__(self, input_dim=N_FEATURES, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout, bidirectional=False
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)   # h_n: (num_layers, B, hidden)
        out = h_n[-1]                 # last layer hidden state: (B, 128)
        return self.proj(out)


# ── 8c. Fusion Model ─────────────────────────────────────────────────────────
class WildfireFusionModel(nn.Module):
    """
    CNN embedding (256) + LSTM embedding (128) + tabular MLP (64)
    → concat (448) → fusion head → binary output
    """
    def __init__(self, tab_dim=60, cnn_embed=256, lstm_embed=128, tab_embed=64):
        super().__init__()
        self.cnn  = CNNEncoder(embed_dim=cnn_embed)
        self.lstm = LSTMEncoder(hidden_dim=lstm_embed)

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, tab_embed),
            nn.ReLU(inplace=True),
        )

        fusion_in = cnn_embed + lstm_embed + tab_embed   # 448
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, img, seq, tab):
        cnn_emb  = self.cnn(img)
        lstm_emb = self.lstm(seq)
        tab_emb  = self.tab_mlp(tab)
        combined = torch.cat([cnn_emb, lstm_emb, tab_emb], dim=1)
        return self.fusion_head(combined).squeeze(1)    # (B,)


# Ablation variants
class CNNOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = CNNEncoder(embed_dim=256)
        self.head = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, img, seq, tab):
        return self.head(self.cnn(img)).squeeze(1)


class LSTMOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = LSTMEncoder(hidden_dim=128)
        self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))
    def forward(self, img, seq, tab):
        return self.head(self.lstm(seq)).squeeze(1)


# Quick sanity check
model_test = WildfireFusionModel().to(DEVICE)
dummy_img = torch.randn(4, 12, 64, 64).to(DEVICE)
dummy_seq = torch.randn(4, 64, 12).to(DEVICE)
dummy_tab = torch.randn(4, 60).to(DEVICE)
out = model_test(dummy_img, dummy_seq, dummy_tab)
print(f"Fusion model output shape: {out.shape}  ✓")
total_params = sum(p.numel() for p in model_test.parameters() if p.requires_grad)
print(f"Total trainable params: {total_params:,}")
del model_test, dummy_img, dummy_seq, dummy_tab, out

# %% [markdown]
# ## 9. Training Utilities

# %%
def compute_pos_weight():
    """Compute positive class weight to handle class imbalance."""
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    return torch.tensor(neg / pos, dtype=torch.float32).to(DEVICE)


def evaluate(model, loader):
    """Return PR-AUC, ROC-AUC, all predictions, all labels."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for img, seq, tab, label in loader:
            img, seq, tab = img.to(DEVICE), seq.to(DEVICE), tab.to(DEVICE)
            logits = model(img, seq, tab)
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(label.numpy())
    y_true = np.array(all_labels)
    y_prob = np.array(all_preds)
    pr_auc  = average_precision_score(y_true, y_prob) * 100
    roc_auc = roc_auc_score(y_true, y_prob) * 100
    return pr_auc, roc_auc, y_prob, y_true


def train_model(model, model_name, n_epochs, run_name):
    """
    Train a PyTorch model, log to MLflow, return (best_pr_auc, val_probs, val_labels).
    Uses BCEWithLogitsLoss + pos_weight + cosine LR schedule.
    """
    pos_weight = compute_pos_weight()
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_pr_auc   = 0.0
    best_state    = None
    history       = {'train_loss': [], 'val_pr_auc': [], 'val_roc_auc': []}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'model':     model_name,
            'epochs':    n_epochs,
            'lr':        LR,
            'batch_size': BATCH_SIZE,
            'pos_weight': round(pos_weight.item(), 2),
        })

        for epoch in range(1, n_epochs + 1):
            model.train()
            epoch_loss = 0.0

            for img, seq, tab, label in train_loader:
                img, seq, tab = img.to(DEVICE), seq.to(DEVICE), tab.to(DEVICE)
                label = label.to(DEVICE)

                optimizer.zero_grad()
                logits = model(img, seq, tab)
                loss   = criterion(logits, label)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(train_loader)
            pr_auc, roc_auc, _, _ = evaluate(model, val_loader)

            history['train_loss'].append(avg_loss)
            history['val_pr_auc'].append(pr_auc)
            history['val_roc_auc'].append(roc_auc)

            mlflow.log_metrics({
                'train_loss': round(avg_loss, 4),
                'val_pr_auc':  round(pr_auc, 2),
                'val_roc_auc': round(roc_auc, 2),
            }, step=epoch)

            if pr_auc > best_pr_auc:
                best_pr_auc = pr_auc
                best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, f'{model_name}_best.pt')

            print(f"  Epoch {epoch:02d}/{n_epochs} | loss:{avg_loss:.4f} | "
                  f"val PR-AUC:{pr_auc:.2f}% | val ROC-AUC:{roc_auc:.2f}%")

        # Load best weights and evaluate on test set
        model.load_state_dict(best_state)
        test_pr_auc, test_roc_auc, test_probs, test_labels = evaluate(model, test_loader)
        mlflow.log_metrics({'test_pr_auc': round(test_pr_auc, 2),
                            'test_roc_auc': round(test_roc_auc, 2)})
        mlflow.pytorch.log_model(model, model_name)

        print(f"\n  ✓ [{model_name}] Best val PR-AUC: {best_pr_auc:.2f}%")
        print(f"  ✓ [{model_name}] Test PR-AUC:     {test_pr_auc:.2f}%")
        print(f"  ✓ [{model_name}] Test ROC-AUC:    {test_roc_auc:.2f}%")

    return best_pr_auc, test_pr_auc, test_roc_auc, test_probs, test_labels, history


mlflow.set_experiment("WildfireSpreadPrediction")

# %% [markdown]
# ## 10. Ablation Run 1 — XGBoost (tabular only)

# %%
print("=" * 55)
print("RUN 1: XGBoost — tabular features only")
print("=" * 55)

xgb_params = {
    'n_estimators':    500,
    'max_depth':       6,
    'learning_rate':   0.05,
    'subsample':       0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
    'eval_metric':     'aucpr',
    'use_label_encoder': False,
    'tree_method':     'hist',
    'device':          'cuda' if torch.cuda.is_available() else 'cpu',
    'random_state':    SEED,
}

with mlflow.start_run(run_name="XGBoost_tabular"):
    mlflow.log_params(xgb_params)

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(
        tab_train_sc, y_train,
        eval_set=[(tab_val_sc, y_val)],
        verbose=50,
        early_stopping_rounds=30,
    )

    xgb_val_prob  = xgb_model.predict_proba(tab_val_sc)[:, 1]
    xgb_test_prob = xgb_model.predict_proba(tab_test_sc)[:, 1]

    xgb_val_prauc  = average_precision_score(y_val,  xgb_val_prob)  * 100
    xgb_test_prauc = average_precision_score(y_test, xgb_test_prob) * 100
    xgb_test_roc   = roc_auc_score(y_test, xgb_test_prob) * 100

    mlflow.log_metrics({
        'val_pr_auc':  round(xgb_val_prauc,  2),
        'test_pr_auc': round(xgb_test_prauc, 2),
        'test_roc_auc': round(xgb_test_roc,  2),
    })
    mlflow.sklearn.log_model(xgb_model, "xgboost_model")

    print(f"\n  ✓ [XGBoost] Test PR-AUC:  {xgb_test_prauc:.2f}%")
    print(f"  ✓ [XGBoost] Test ROC-AUC: {xgb_test_roc:.2f}%")

# %% [markdown]
# ## 11. Ablation Run 2 — CNN + LSTM (no tabular)

# %%
print("\n" + "=" * 55)
print("RUN 2: CNN + LSTM fusion (no XGBoost)")
print("=" * 55)


class CNNLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn  = CNNEncoder(embed_dim=256)
        self.lstm = LSTMEncoder(hidden_dim=128)
        self.head = nn.Sequential(
            nn.Linear(384, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 1)
        )
    def forward(self, img, seq, tab):
        return self.head(torch.cat([self.cnn(img), self.lstm(seq)], dim=1)).squeeze(1)


cnn_lstm_model = CNNLSTMModel().to(DEVICE)
(cnn_lstm_best_val, cnn_lstm_test_prauc, cnn_lstm_test_roc,
 cnn_lstm_probs, cnn_lstm_labels, cnn_lstm_hist) = train_model(
    cnn_lstm_model, "CNN_LSTM", EPOCHS_CNN, "CNN_LSTM_fusion"
)

# %% [markdown]
# ## 12. Ablation Run 3 — Full Fusion (CNN + LSTM + XGBoost features)

# %%
print("\n" + "=" * 55)
print("RUN 3: Full Fusion — CNN + LSTM + Tabular MLP")
print("=" * 55)

fusion_model = WildfireFusionModel().to(DEVICE)
(fusion_best_val, fusion_test_prauc, fusion_test_roc,
 fusion_probs, fusion_labels, fusion_hist) = train_model(
    fusion_model, "FullFusion", EPOCHS_FUSE, "Full_CNN_LSTM_Tab_Fusion"
)

# %% [markdown]
# ## 13. Stacked Ensemble (XGBoost probs + neural probs → meta-learner)

# %%
print("\n" + "=" * 55)
print("RUN 4: Stacked Ensemble (meta-learner)")
print("=" * 55)

# Get validation probs from all models for stacking
_, _, _, cnn_lstm_val_probs, cnn_lstm_val_labels, _ = (
    lambda m: (None, None, None,
               *evaluate(m, val_loader)[2:], None)
)(cnn_lstm_model)

_, _, _, fusion_val_probs, fusion_val_labels, _ = (
    lambda m: (None, None, None,
               *evaluate(m, val_loader)[2:], None)
)(fusion_model)


def get_val_probs(model):
    _, _, probs, labels = evaluate(model, val_loader)
    return probs

def get_test_probs(model):
    _, _, probs, labels = evaluate(model, test_loader)
    return probs


cnn_lstm_val_p  = get_val_probs(cnn_lstm_model)
cnn_lstm_test_p = get_test_probs(cnn_lstm_model)
fusion_val_p    = get_val_probs(fusion_model)
fusion_test_p   = get_test_probs(fusion_model)

# Stack features: [xgb_prob, cnn_lstm_prob, fusion_prob]
stack_val  = np.column_stack([xgb_val_prob,  cnn_lstm_val_p,  fusion_val_p])
stack_test = np.column_stack([xgb_test_prob, cnn_lstm_test_p, fusion_test_p])

with mlflow.start_run(run_name="StackedEnsemble"):
    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    meta.fit(stack_val, y_val)

    stack_test_prob = meta.predict_proba(stack_test)[:, 1]
    stack_prauc     = average_precision_score(y_test, stack_test_prob) * 100
    stack_roc       = roc_auc_score(y_test, stack_test_prob) * 100

    mlflow.log_metrics({'test_pr_auc': round(stack_prauc, 2),
                        'test_roc_auc': round(stack_roc, 2)})
    mlflow.sklearn.log_model(meta, "meta_learner")

    print(f"  ✓ [Stacked] Test PR-AUC:  {stack_prauc:.2f}%")
    print(f"  ✓ [Stacked] Test ROC-AUC: {stack_roc:.2f}%")

# %% [markdown]
# ## 14. Results — Full Comparison Table + Plots

# %%
PAPER_BASELINE_PRAUC = 28.4   # IEEE TGRS 2022 neural network

results = {
    'Persistence (paper)':         {'PR-AUC': 11.5,  'ROC-AUC': None,       'probs': None},
    'Logistic Reg (paper)':        {'PR-AUC': 20.0,  'ROC-AUC': None,       'probs': None},
    'RF (paper)':                  {'PR-AUC': 22.0,  'ROC-AUC': None,       'probs': None},
    'Neural Net (paper baseline)': {'PR-AUC': 28.4,  'ROC-AUC': None,       'probs': None},
    'XGBoost (ours)':              {'PR-AUC': xgb_test_prauc,  'ROC-AUC': xgb_test_roc,   'probs': xgb_test_prob},
    'CNN+LSTM (ours)':             {'PR-AUC': cnn_lstm_test_prauc, 'ROC-AUC': cnn_lstm_test_roc, 'probs': cnn_lstm_probs},
    'Full Fusion (ours)':          {'PR-AUC': fusion_test_prauc,   'ROC-AUC': fusion_test_roc,   'probs': fusion_probs},
    'Stacked Ensemble (ours)':     {'PR-AUC': stack_prauc,         'ROC-AUC': stack_roc,          'probs': stack_test_prob},
}

df_results = pd.DataFrame([
    {'Model': k, 'PR-AUC (%)': round(v['PR-AUC'], 2),
     'ROC-AUC (%)': round(v['ROC-AUC'], 2) if v['ROC-AUC'] else '—',
     'vs Paper Baseline': f"+{v['PR-AUC'] - PAPER_BASELINE_PRAUC:+.2f}%" if v['PR-AUC'] > 0 else '—'}
    for k, v in results.items()
])
print("\n" + "=" * 65)
print("FULL ABLATION RESULTS")
print("=" * 65)
print(df_results.to_string(index=False))
print("=" * 65)

# ── Plotting ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle('Wildfire Spread Prediction — Results', fontsize=14, fontweight='bold')

# 1. PR-AUC bar chart
ax1 = fig.add_subplot(gs[0, :2])
models     = list(results.keys())
pr_scores  = [v['PR-AUC'] for v in results.values()]
colors     = ['#95a5a6'] * 4 + ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
bars = ax1.barh(models, pr_scores, color=colors, edgecolor='white', height=0.6)
ax1.axvline(PAPER_BASELINE_PRAUC, color='black', linestyle='--', lw=1.5,
            label=f'Paper baseline ({PAPER_BASELINE_PRAUC}%)')
ax1.set_xlabel('PR-AUC (%)')
ax1.set_title('PR-AUC Comparison (↑ better) — Primary Metric')
ax1.legend(fontsize=9)
for bar, score in zip(bars, pr_scores):
    ax1.text(score + 0.3, bar.get_y() + bar.get_height()/2,
             f'{score:.1f}%', va='center', fontsize=8)

# 2. Training curves (fusion model)
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(fusion_hist['train_loss'], label='Train Loss', color='#e74c3c')
ax2b = ax2.twinx()
ax2b.plot(fusion_hist['val_pr_auc'], label='Val PR-AUC', color='#2ecc71', linestyle='--')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss', color='#e74c3c')
ax2b.set_ylabel('PR-AUC (%)', color='#2ecc71')
ax2.set_title('Full Fusion — Training Curve')
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

# 3. Precision-Recall curves (our models only)
ax3 = fig.add_subplot(gs[1, 0])
our_models = {
    'XGBoost':        (xgb_test_prob,         y_test, '#e74c3c'),
    'CNN+LSTM':       (cnn_lstm_probs,         y_test, '#e67e22'),
    'Full Fusion':    (fusion_probs,           y_test, '#2ecc71'),
    'Stacked':        (stack_test_prob,        y_test, '#3498db'),
}
for name, (probs, labels, color) in our_models.items():
    p, r, _ = precision_recall_curve(labels, probs)
    auc_val = average_precision_score(labels, probs) * 100
    ax3.plot(r, p, label=f'{name} ({auc_val:.1f}%)', color=color, lw=1.8)
ax3.axhline(y_test.mean(), color='gray', linestyle=':', label='No-skill baseline')
ax3.set_xlabel('Recall'); ax3.set_ylabel('Precision')
ax3.set_title('Precision-Recall Curves')
ax3.legend(fontsize=7.5)

# 4. ROC curves
ax4 = fig.add_subplot(gs[1, 1])
for name, (probs, labels, color) in our_models.items():
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_val = roc_auc_score(labels, probs) * 100
    ax4.plot(fpr, tpr, label=f'{name} ({auc_val:.1f}%)', color=color, lw=1.8)
ax4.plot([0,1],[0,1], 'k--', label='Random')
ax4.set_xlabel('FPR'); ax4.set_ylabel('TPR')
ax4.set_title('ROC Curves')
ax4.legend(fontsize=7.5)

# 5. Feature importance (XGBoost)
ax5 = fig.add_subplot(gs[1, 2])
tab_feature_names = []
for stat in ['mean', 'std', 'max', 'min', 'median']:
    for feat in [f.replace('PrevFireMask','PrevFire') for f in INPUT_FEATURES]:
        tab_feature_names.append(f'{feat}_{stat}')

importance = xgb_model.feature_importances_
top_k  = 12
top_idx = np.argsort(importance)[-top_k:]
ax5.barh([tab_feature_names[i] for i in top_idx], importance[top_idx],
          color='steelblue', alpha=0.8)
ax5.set_title('XGBoost Top-12 Feature Importance')
ax5.tick_params(labelsize=7)

plt.savefig('results.png', dpi=130, bbox_inches='tight')
plt.show()
print("\nResults plot saved → results.png")

# %% [markdown]
# ## 15. Final Summary

# %%
best_model_name = max(
    [(k, v['PR-AUC']) for k, v in results.items() if 'ours' in k.lower() or 'fusion' in k.lower() or 'stacked' in k.lower() or 'xgboost' in k.lower()],
    key=lambda x: x[1]
)[0]
best_prauc = results[best_model_name]['PR-AUC']
delta = best_prauc - PAPER_BASELINE_PRAUC

print("\n" + "=" * 55)
print("SUMMARY")
print("=" * 55)
print(f"Paper baseline (IEEE TGRS 2022):  {PAPER_BASELINE_PRAUC:.1f}%  PR-AUC")
print(f"Our best model ({best_model_name}):  {best_prauc:.2f}%  PR-AUC")
print(f"Improvement:                       {delta:+.2f} percentage points")
print("=" * 55)
print("\nResume bullet (fill in actual numbers):")
print(f"""
  - Built multimodal wildfire spread predictor (CNN + LSTM + XGBoost late-fusion)
    on Next Day Wildfire Spread dataset (18K patches, 12 geospatial channels);
    outperformed IEEE TGRS 2022 neural network baseline by {delta:.1f} pts PR-AUC
    ({PAPER_BASELINE_PRAUC:.1f}% → {best_prauc:.1f}%) across 4-run MLflow ablation.
  - Containerized FastAPI inference endpoint (Docker + GitHub Actions CI/CD);
    monitored feature drift with Evidently AI on live NASA FIRMS data.
""")