import numpy as np
import matplotlib.pyplot as plt
import os

SAVE_DIR = os.path.expanduser("~/Desktop/NPB_GAN/data/processed")

epochs = np.load(os.path.join(SAVE_DIR, "epochs.npy"))
labels = np.load(os.path.join(SAVE_DIR, "labels.npy"))

# ── Basic statistics ───────────────────────────────
print("── Value range check ─────────────────────────────")
print(f"  Min value : {epochs.min():.4f}")
print(f"  Max value : {epochs.max():.4f}")
print(f"  Mean      : {epochs.mean():.4f}")
print(f"  Std dev   : {epochs.std():.4f}")
print(f"  Any NaNs? : {np.isnan(epochs).any()}")
print(f"  Any Infs? : {np.isinf(epochs).any()}")

# ── Plot one epoch of each class ───────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
class_names = {0: "Interictal", 1: "Preictal", 2: "Ictal"}
colors      = {0: "steelblue", 1: "orange", 2: "crimson"}

for label_val in [0, 1, 2]:
    idx   = np.where(labels == label_val)[0][0]  # first epoch of this class
    epoch = epochs[idx, 0, :]                     # channel 0 only
    ax    = axes[label_val]
    ax.plot(epoch, color=colors[label_val], linewidth=0.8)
    ax.set_title(f"{class_names[label_val]} example (epoch {idx}, channel 0)")
    ax.set_xlabel("Timepoints")
    ax.set_ylabel("Amplitude (V)")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "sanity_check.png"), dpi=150)
print("\n  Plot saved to data/processed/sanity_check.png")
print("  Open it in Finder to visually inspect the signals.")