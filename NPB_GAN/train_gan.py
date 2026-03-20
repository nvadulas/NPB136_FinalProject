import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.expanduser("~/Desktop/NPB_GAN/data/processed")
MODELS_DIR    = os.path.expanduser("~/Desktop/NPB_GAN/models")
os.makedirs(MODELS_DIR, exist_ok=True)

N_CHANNELS  = 6
N_BANDS     = 5
N_FEATURES  = N_CHANNELS * N_BANDS   # 30 — flattened input size
N_CLASSES   = 3
LATENT_DIM  = 64
EMBED_DIM   = 16
BATCH_SIZE  = 256
N_EPOCHS    = 500
LR          = 0.0002
SAVE_EVERY  = 100

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print(f"Using device: {DEVICE}")


# ── Normalize ──────────────────────────────────────────────────────────────────
# Band power values are always positive and can be very large
# Log transform first, then normalize to [-1, 1]

def normalize(epochs):
    # epochs shape: (N, 6, 5)
    epochs = np.log1p(epochs)                    # log(1+x) handles zeros safely
    flat   = epochs.reshape(len(epochs), -1)     # (N, 30)
    mean   = flat.mean(axis=0)
    std    = flat.std(axis=0) + 1e-8
    flat   = (flat - mean) / std                 # standardize each feature
    np.save(os.path.join(PROCESSED_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(PROCESSED_DIR, "norm_std.npy"),  std)
    return flat.astype(np.float32)


# ── Balance ────────────────────────────────────────────────────────────────────

def balance_dataset(epochs_flat, labels):
    idx_ictal      = np.where(labels == 2)[0]
    idx_preictal   = np.where(labels == 1)[0]
    idx_interictal = np.where(labels == 0)[0]

    n_ictal = len(idx_ictal)   # 5844
    np.random.seed(42)

    idx_preictal   = np.random.choice(idx_preictal,   size=n_ictal, replace=False)
    idx_interictal = np.random.choice(idx_interictal, size=n_ictal, replace=False)

    all_idx = np.concatenate([idx_ictal, idx_preictal, idx_interictal])
    np.random.shuffle(all_idx)

    print(f"  Balanced dataset: {n_ictal} per class, {len(all_idx)} total")
    return epochs_flat[all_idx], labels[all_idx]


# ── Load data ──────────────────────────────────────────────────────────────────

def load_data():
    epochs = np.load(os.path.join(PROCESSED_DIR, "epochs.npy"))
    labels = np.load(os.path.join(PROCESSED_DIR, "labels.npy")).astype(np.int64)

    epochs_flat        = normalize(epochs)
    epochs_bal, labels_bal = balance_dataset(epochs_flat, labels)

    X = torch.tensor(epochs_bal)
    y = torch.tensor(labels_bal)

    return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE, shuffle=True)


# ── Generator ─────────────────────────────────────────────────────────────────
# Input:  noise (64,) + class label → Output: fake feature vector (30,)
# Simple MLP — no convolutions needed for 30 features

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM + EMBED_DIM, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, N_FEATURES),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_vec = self.label_embed(labels)
        x = torch.cat([noise, label_vec], dim=1)
        return self.net(x)   # (batch, 30)


# ── Discriminator ─────────────────────────────────────────────────────────────
# Input:  feature vector (30,) + class label → Output: real/fake score

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_embed = nn.Embedding(N_CLASSES, EMBED_DIM)
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES + EMBED_DIM, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        label_vec = self.label_embed(labels)
        x = torch.cat([x, label_vec], dim=1)
        return self.net(x)   # (batch, 1)


# ── Training ───────────────────────────────────────────────────────────────────

def train():
    dataloader = load_data()

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR,       betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR * 0.5, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    print(f"\n── Training for {N_EPOCHS} epochs ────────────────────────")
    print(f"   Input features : {N_FEATURES} (6 channels × 5 bands)")
    print(f"   Batch size     : {BATCH_SIZE}")
    print(f"   Device         : {DEVICE}\n")

    for epoch in range(1, N_EPOCHS + 1):
        d_losses = []
        g_losses = []

        for real_x, real_labels in dataloader:
            real_x      = real_x.to(DEVICE)
            real_labels = real_labels.to(DEVICE)
            batch_size  = real_x.size(0)

            real_targets = torch.full((batch_size, 1), 0.9).to(DEVICE)
            fake_targets = torch.zeros(batch_size, 1).to(DEVICE)

            # ── Train Discriminator ──────────────────────────────
            D.zero_grad()
            d_real      = D(real_x, real_labels)
            d_loss_real = criterion(d_real, real_targets)

            noise       = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
            fake_labels = torch.randint(0, N_CLASSES, (batch_size,)).to(DEVICE)
            fake_x      = G(noise, fake_labels).detach()
            d_fake      = D(fake_x, fake_labels)
            d_loss_fake = criterion(d_fake, fake_targets)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_D.step()

            # ── Train Generator twice ────────────────────────────
            for _ in range(2):
                G.zero_grad()
                noise       = torch.randn(batch_size, LATENT_DIM).to(DEVICE)
                fake_labels = torch.randint(0, N_CLASSES, (batch_size,)).to(DEVICE)
                fake_x      = G(noise, fake_labels)
                d_pred      = D(fake_x, fake_labels)
                g_loss      = criterion(d_pred, real_targets)
                g_loss.backward()
                opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:4d}/{N_EPOCHS} | "
                  f"D loss: {np.mean(d_losses):.4f} | "
                  f"G loss: {np.mean(g_losses):.4f}")

        if epoch % SAVE_EVERY == 0:
            torch.save(G.state_dict(), os.path.join(MODELS_DIR, f"generator_epoch{epoch}.pt"))
            torch.save(D.state_dict(), os.path.join(MODELS_DIR, f"discriminator_epoch{epoch}.pt"))
            print(f"  ✓ Checkpoint saved at epoch {epoch}")

    torch.save(G.state_dict(), os.path.join(MODELS_DIR, "generator_final.pt"))
    torch.save(D.state_dict(), os.path.join(MODELS_DIR, "discriminator_final.pt"))
    print("\n  ✓ Training complete!")


if __name__ == "__main__":
    train()