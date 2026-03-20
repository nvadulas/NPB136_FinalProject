import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Configuration ─────────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.expanduser("~/Desktop/NPB_GAN/data/processed")
MODELS_DIR    = os.path.expanduser("~/Desktop/NPB_GAN/models")
OUTPUTS_DIR   = os.path.expanduser("~/Desktop/NPB_GAN/outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

N_FEATURES  = 30
N_CLASSES   = 3
LATENT_DIM  = 64
EMBED_DIM   = 16
N_SYNTHETIC = 2000   # per class

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

BAND_NAMES    = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
CHANNEL_NAMES = ["FP1-F7", "F7-T7", "T7-P7", "FP1-F3", "F3-C3", "C3-P3"]
CLASS_NAMES   = {0: "Interictal", 1: "Preictal", 2: "Ictal"}
COLORS        = {0: "steelblue", 1: "darkorange", 2: "crimson"}


# ── Generator (must match train_gan.py exactly) ───────────────────────────────

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
        return self.net(x)


# ── Load generator ─────────────────────────────────────────────────────────────

def load_generator():
    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, "generator_final.pt"),
        map_location=DEVICE
    ))
    G.eval()
    print("  Generator loaded.")
    return G


# ── Denormalize ────────────────────────────────────────────────────────────────
# Reverse the log + standardization we applied during preprocessing

def denormalize(flat):
    mean  = np.load(os.path.join(PROCESSED_DIR, "norm_mean.npy"))
    std   = np.load(os.path.join(PROCESSED_DIR, "norm_std.npy"))
    flat  = flat * std + mean          # reverse standardization
    flat  = np.expm1(flat)             # reverse log1p
    flat  = np.clip(flat, 0, None)     # band power can't be negative
    return flat


# ── Generate synthetic samples ─────────────────────────────────────────────────

def generate_synthetic(G, label_val, n_samples):
    all_samples = []
    batch_size  = 256

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            count  = min(batch_size, n_samples - start)
            noise  = torch.randn(count, LATENT_DIM).to(DEVICE)
            labels = torch.full((count,), label_val, dtype=torch.long).to(DEVICE)
            fake   = G(noise, labels).cpu().numpy()
            all_samples.append(fake)

    samples_flat = np.concatenate(all_samples, axis=0)   # (N, 30) normalized
    samples_real = denormalize(samples_flat)              # back to band power
    return samples_real.reshape(-1, 6, 5)                 # (N, 6, 5)


# ── Plot: mean band power per class ───────────────────────────────────────────

def plot_band_profiles(real_epochs, real_labels, synthetic_by_class):
    """
    For each frequency band, plot the mean power across classes
    for both real and synthetic data side by side.
    """
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    fig.suptitle("Mean band power by class — Real vs Synthetic", fontsize=13)

    for band_idx, band_name in enumerate(BAND_NAMES):
        ax = axes[band_idx]

        for label_val in [0, 1, 2]:
            # Real: mean across all channels for this band
            real_mask  = real_labels == label_val
            real_power = real_epochs[real_mask, :, band_idx].mean(axis=1)
            real_mean  = real_power.mean()

            # Synthetic
            syn_power  = synthetic_by_class[label_val][:, :, band_idx].mean(axis=1)
            syn_mean   = syn_power.mean()

            color = COLORS[label_val]
            name  = CLASS_NAMES[label_val]

            ax.bar(label_val * 2,     real_mean, color=color, alpha=0.9,
                   label=f"{name} real",      width=0.8)
            ax.bar(label_val * 2 + 1, syn_mean,  color=color, alpha=0.4,
                   label=f"{name} synthetic", width=0.8, hatch="//")

        ax.set_title(f"{band_name} band")
        ax.set_xlabel("Class (solid=real, hatched=synthetic)")
        ax.set_ylabel("Mean power (μV²/Hz)")
        ax.set_xticks([0.5, 2.5, 4.5])
        ax.set_xticklabels(["Interictal", "Preictal", "Ictal"], fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUTS_DIR, "band_power_comparison.png")
    plt.savefig(path, dpi=150)
    print(f"  Plot saved to {path}")
    plt.close()


# ── Save augmented dataset ─────────────────────────────────────────────────────

def save_augmented(real_epochs, real_labels, synthetic_by_class):
    syn_epochs = []
    syn_labels = []

    for label_val, epochs in synthetic_by_class.items():
        syn_epochs.append(epochs)
        syn_labels.append(np.full(len(epochs), label_val, dtype=np.int64))

    syn_epochs = np.concatenate(syn_epochs, axis=0)
    syn_labels = np.concatenate(syn_labels, axis=0)

    combined_epochs = np.concatenate([real_epochs, syn_epochs], axis=0)
    combined_labels = np.concatenate([real_labels, syn_labels], axis=0)

    idx = np.random.permutation(len(combined_labels))
    combined_epochs = combined_epochs[idx]
    combined_labels = combined_labels[idx]

    np.save(os.path.join(PROCESSED_DIR, "epochs_augmented.npy"), combined_epochs)
    np.save(os.path.join(PROCESSED_DIR, "labels_augmented.npy"), combined_labels)

    print(f"\n── Augmented dataset ─────────────────────────────────")
    print(f"  Real       : {len(real_labels)}")
    print(f"  Synthetic  : {len(syn_labels)}")
    print(f"  Total      : {len(combined_labels)}")
    print(f"  Interictal : {np.sum(combined_labels == 0)}")
    print(f"  Preictal   : {np.sum(combined_labels == 1)}")
    print(f"  Ictal      : {np.sum(combined_labels == 2)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    real_epochs = np.load(os.path.join(PROCESSED_DIR, "epochs.npy"))
    real_labels = np.load(os.path.join(PROCESSED_DIR, "labels.npy")).astype(np.int64)

    # Convert real epochs to band power for fair comparison
    # (they're already in (N, 6, 5) shape from preprocessing)
    print(f"  Real data shape: {real_epochs.shape}")

    G = load_generator()

    print(f"\n  Generating {N_SYNTHETIC} synthetic samples per class...")
    synthetic_by_class = {}
    for label_val in [0, 1, 2]:
        synthetic_by_class[label_val] = generate_synthetic(G, label_val, N_SYNTHETIC)
        print(f"  {CLASS_NAMES[label_val]}: {len(synthetic_by_class[label_val])} samples")

    print("\n  Plotting band power comparison...")
    plot_band_profiles(real_epochs, real_labels, synthetic_by_class)

    print("\n  Saving augmented dataset...")
    save_augmented(real_epochs, real_labels, synthetic_by_class)

    print("\n  Done! Open outputs/band_power_comparison.png to inspect quality.")


if __name__ == "__main__":
    main()