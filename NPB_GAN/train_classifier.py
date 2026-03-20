import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.expanduser("~/Desktop/NPB_GAN/data/processed")
MODELS_DIR    = os.path.expanduser("~/Desktop/NPB_GAN/models")
OUTPUTS_DIR   = os.path.expanduser("~/Desktop/NPB_GAN/outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

N_CLASSES   = 3
N_FEATURES  = 30      # 6 channels × 5 bands, flattened
BATCH_SIZE  = 256
N_EPOCHS    = 150
LR          = 0.0005

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {DEVICE}")

CLASS_NAMES = ["Interictal", "Preictal", "Ictal"]


# ── Normalize ──────────────────────────────────────────────────────────────────

def normalize(epochs):
    flat = np.log1p(epochs).reshape(len(epochs), -1).astype(np.float32)
    mean = flat.mean(axis=0)
    std  = flat.std(axis=0) + 1e-8
    return (flat - mean) / std


# ── Load datasets ──────────────────────────────────────────────────────────────

def load_real():
    epochs = np.load(os.path.join(PROCESSED_DIR, "epochs.npy"))
    labels = np.load(os.path.join(PROCESSED_DIR, "labels.npy")).astype(np.int64)
    print(f"  Real data: {len(labels)} epochs")
    print(f"    Interictal: {np.sum(labels==0)}")
    print(f"    Preictal  : {np.sum(labels==1)}")
    print(f"    Ictal     : {np.sum(labels==2)}")
    return normalize(epochs), labels


def load_augmented():
    epochs = np.load(os.path.join(PROCESSED_DIR, "epochs_augmented.npy"))
    labels = np.load(os.path.join(PROCESSED_DIR, "labels_augmented.npy")).astype(np.int64)
    print(f"  Augmented data: {len(labels)} epochs")
    print(f"    Interictal: {np.sum(labels==0)}")
    print(f"    Preictal  : {np.sum(labels==1)}")
    print(f"    Ictal     : {np.sum(labels==2)}")
    return normalize(epochs), labels


# ── Classifier model ───────────────────────────────────────────────────────────
# Simple but effective MLP for tabular EEG features

class EEGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, N_CLASSES)
            # No softmax here — CrossEntropyLoss handles that internally
        )

    def forward(self, x):
        return self.net(x)


# ── Train one model ────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val, label, class_weights):
    print(f"\n── Training {label} ───────────────────────────────────")

    X_tr = torch.tensor(X_train)
    y_tr = torch.tensor(y_train)
    X_v  = torch.tensor(X_val)
    y_v  = torch.tensor(y_val)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model     = EEGClassifier().to(DEVICE)

    # Class weights help with imbalance — ictal gets higher weight
    weights   = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    train_losses = []
    val_losses   = []
    val_accs     = []
    best_val_acc = 0
    best_state   = None

    for epoch in range(1, N_EPOCHS + 1):
        # ── Training pass ──────────────────────────────────────
        model.train()
        batch_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)

        # ── Validation pass ────────────────────────────────────
        model.eval()
        with torch.no_grad():
            X_v_dev  = X_v.to(DEVICE)
            y_v_dev  = y_v.to(DEVICE)
            val_pred = model(X_v_dev)
            val_loss = criterion(val_pred, y_v_dev).item()
            val_acc  = (val_pred.argmax(dim=1) == y_v_dev).float().mean().item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{N_EPOCHS} | "
                  f"Train loss: {train_loss:.4f} | "
                  f"Val loss: {val_loss:.4f} | "
                  f"Val acc: {val_acc*100:.1f}%")

    print(f"\n  Best validation accuracy: {best_val_acc*100:.1f}%")

    # Save best model weights
    model_path = os.path.join(MODELS_DIR, f"classifier_{label.replace(' ', '_')}.pt")
    torch.save(best_state, model_path)
    print(f"  Model saved to {model_path}")

    return model, best_state, train_losses, val_losses, val_accs


# ── Evaluate ───────────────────────────────────────────────────────────────────

def evaluate(model, best_state, X_test, y_test, label):
    print(f"\n── Evaluation: {label} ────────────────────────────────")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        X_t   = torch.tensor(X_test).to(DEVICE)
        preds = model(X_t).argmax(dim=1).cpu().numpy()

    print(classification_report(y_test, preds, target_names=CLASS_NAMES))

    # Confusion matrix
    cm  = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax
    )
    ax.set_title(f"Confusion matrix — {label}")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, f"confusion_{label.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {path}")

    return preds


# ── Plot training curves ───────────────────────────────────────────────────────

def plot_curves(losses_a, losses_b, accs_a, accs_b):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    axes[0].plot(losses_a, label="Real only",      color="steelblue")
    axes[0].plot(losses_b, label="Real+Synthetic", color="darkorange")
    axes[0].set_title("Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy curves
    axes[1].plot([a*100 for a in accs_a], label="Real only",      color="steelblue")
    axes[1].plot([a*100 for a in accs_b], label="Real+Synthetic", color="darkorange")
    axes[1].set_title("Validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()

    plt.suptitle("Classifier A (real only) vs Classifier B (augmented)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Training curves saved to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── Load data ──────────────────────────────────────────────
    print("── Loading data ──────────────────────────────────────")
    X_real, y_real = load_real()
    X_aug,  y_aug  = load_augmented()

    # ── Train/val/test split ───────────────────────────────────
    # Use the same test set for both classifiers for fair comparison
    # 70% train, 15% val, 15% test
    X_real_tv, X_test, y_real_tv, y_test = train_test_split(
        X_real, y_real, test_size=0.15, random_state=42, stratify=y_real
    )
    X_real_tr, X_val, y_real_tr, y_val = train_test_split(
        X_real_tv, y_real_tv, test_size=0.176, random_state=42, stratify=y_real_tv
    )
    # 0.176 of 0.85 ≈ 0.15 of total

    # Augmented training set (same val and test as above for fair comparison)
    X_aug_tv, _, y_aug_tv, _ = train_test_split(
        X_aug, y_aug, test_size=0.15, random_state=42, stratify=y_aug
    )
    X_aug_tr, _, y_aug_tr, _ = train_test_split(
        X_aug_tv, y_aug_tv, test_size=0.176, random_state=42, stratify=y_aug_tv
    )

    print(f"\n  Train (real)     : {len(y_real_tr)}")
    print(f"  Train (augmented): {len(y_aug_tr)}")
    print(f"  Val              : {len(y_val)}")
    print(f"  Test             : {len(y_test)}")

    # ── Class weights ──────────────────────────────────────────
    # Give rare classes more weight so the model doesn't just
    # predict interictal all the time
    n_total  = len(y_real_tr)
    n_per    = np.bincount(y_real_tr)
    weights = 1.0 / np.sqrt(n_per)
    weights = weights / weights.sum() * N_CLASSES
    print(f"\n  Class weights: {weights.round(3)}")

    # ── Train both classifiers ─────────────────────────────────
    model_a, state_a, tr_loss_a, val_loss_a, val_acc_a = train_model(
        X_real_tr, y_real_tr, X_val, y_val,
        label="Real only",
        class_weights=weights
    )

    model_b, state_b, tr_loss_b, val_loss_b, val_acc_b = train_model(
        X_aug_tr, y_aug_tr, X_val, y_val,
        label="Real + Synthetic",
        class_weights=weights
    )

    # ── Evaluate on held-out test set ──────────────────────────
    print("\n" + "="*55)
    print("  FINAL TEST SET RESULTS")
    print("="*55)
    evaluate(model_a, state_a, X_test, y_test, "Real only")
    evaluate(model_b, state_b, X_test, y_test, "Real + Synthetic")

    # ── Plot training curves ───────────────────────────────────
    plot_curves(val_loss_a, val_loss_b, val_acc_a, val_acc_b)

    print("\n── All done! Check outputs/ for plots ────────────────")


if __name__ == "__main__":
    main()