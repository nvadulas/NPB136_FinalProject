import os
import numpy as np
import mne
from scipy.signal import welch

# ── Configuration ─────────────────────────────────────────────────────────────

# Path to the downloaded chb01 folder
DATA_DIR = os.path.expanduser(
    "~/Desktop/NPB_GAN/data/raw/physionet.org/files/chbmit/1.0.0"
)
SAVE_DIR = os.path.expanduser("~/Desktop/NPB_GAN/data/processed")

# How long each epoch is in seconds
EPOCH_LENGTH_SEC = 1

# How long before a seizure counts as "preictal" (in seconds)
PREICTAL_WINDOW_SEC = 60 * 30   # 30 minutes

# EEG bandpass filter range (removes noise outside normal brain activity)
FREQ_LOW  = 0.5   # Hz
FREQ_HIGH = 50.0  # Hz

# Channels to use (these 6 are the most commonly used in seizure research)
CHANNELS_TO_USE = [
    "FP1-F7", "F7-T7", "T7-P7",
    "FP1-F3", "F3-C3", "C3-P3"
]

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Parse the summary file ─────────────────────────────────────────────────────
# The summary file tells us which EDF files contain seizures
# and exactly when those seizures start and end (in seconds)

def parse_summary(summary_path):
    seizures = {}
    current_file = None
    current_start = None

    with open(summary_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line.startswith("File Name:"):
            current_file = line.split(": ")[1].strip()
            seizures[current_file] = []

        # Handles both:
        #   "Seizure Start Time: 2996 seconds"    (chb01 style)
        #   "Seizure 1 Start Time: 1724 seconds"  (chb06+ style)
        elif "Start Time:" in line and "Seizure" in line:
            val = line.split("Start Time:")[1].replace("seconds", "").strip()
            try:
                current_start = int(val)
            except ValueError:
                current_start = None

        elif "End Time:" in line and "Seizure" in line:
            val = line.split("End Time:")[1].replace("seconds", "").strip()
            try:
                end = int(val)
                if current_file and current_start is not None:
                    seizures[current_file].append((current_start, end))
                    current_start = None
            except ValueError:
                pass

    return seizures

# ── Label a single epoch ───────────────────────────────────────────────────────

def get_label(epoch_start, epoch_end, seizure_times):
    """
    Given an epoch's start/end time (in seconds) and a list of
    seizure (start, end) tuples, return:
      2 = ictal      (epoch overlaps with a seizure)
      1 = preictal   (epoch is within 30 min before a seizure)
      0 = interictal (everything else)
    """
    for sz_start, sz_end in seizure_times:
        # Ictal: epoch overlaps the seizure window
        if epoch_start < sz_end and epoch_end > sz_start:
            return 2

        # Preictal: epoch falls within the preictal window before seizure
        if sz_start - PREICTAL_WINDOW_SEC <= epoch_start < sz_start:
            return 1

    return 0


def epoch_to_bandpower(epoch, sfreq=256):
    """
    Convert (n_channels, n_timepoints) epoch into
    (n_channels, n_bands) band power features.
    Bands: delta, theta, alpha, beta, gamma
    """
    bands = {
        "delta": (0.5, 4),
        "theta": (4,   8),
        "alpha": (8,  13),
        "beta":  (13, 30),
        "gamma": (30, 50),
    }

    features = []
    for ch in range(epoch.shape[0]):
        freqs, psd = welch(epoch[ch], fs=sfreq, nperseg=min(256, epoch.shape[1]))
        ch_powers  = []
        for band, (lo, hi) in bands.items():
            mask  = (freqs >= lo) & (freqs <= hi)
            power = np.trapezoid(psd[mask], freqs[mask])  # area under PSD curve
            ch_powers.append(power)
        features.append(ch_powers)

    return np.array(features)   # shape: (6, 5)

# ── Process one EDF file ───────────────────────────────────────────────────────

def process_file(edf_path, seizure_times):
    """
    Loads one EDF file, filters it, slices into epochs,
    labels each epoch, and returns arrays + labels.
    """
    print(f"  Processing {os.path.basename(edf_path)} ...")

    # Load the EDF file (MNE handles this natively)
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # Keep only our chosen channels (skip file if any are missing)
    available = raw.ch_names
    channels  = [c for c in CHANNELS_TO_USE if c in available]
    if len(channels) < 2:
        print(f"    Skipping — not enough matching channels.")
        return None, None

    raw.pick_channels(channels)

    # Bandpass filter: removes slow drift and high-frequency noise
    raw.filter(FREQ_LOW, FREQ_HIGH, fir_window="hamming", verbose=False)

    sfreq      = raw.info["sfreq"]           # samples per second (usually 256)
    epoch_samp = int(EPOCH_LENGTH_SEC * sfreq)  # samples per epoch
    data       = raw.get_data()              # shape: (n_channels, n_samples)
    n_samples  = data.shape[1]

    epochs = []
    labels = []

    # Slide through the recording in non-overlapping epochs
    for start_samp in range(0, n_samples - epoch_samp, epoch_samp):
        end_samp   = start_samp + epoch_samp
        start_sec  = start_samp / sfreq
        end_sec    = end_samp   / sfreq

        epoch = data[:, start_samp:end_samp]  # shape: (n_channels, epoch_samp)
        label = get_label(start_sec, end_sec, seizure_times)

        bp = epoch_to_bandpower(epoch, sfreq=int(sfreq))
        epochs.append(bp)

        labels.append(label)

    return np.array(epochs), np.array(labels)



# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    # Auto-detect all downloaded subjects
    subjects = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("chb")
    ])

    print(f"Found {len(subjects)} subjects: {', '.join(subjects)}\n")

    all_epochs = []
    all_labels = []

    for subject in subjects:
        subject_dir  = os.path.join(DATA_DIR, subject)
        summary_path = os.path.join(subject_dir, f"{subject}-summary.txt")

        if not os.path.exists(summary_path):
            print(f"  {subject}: no summary file, skipping\n")
            continue

        seizure_map = parse_summary(summary_path)
        edf_files   = sorted([f for f in os.listdir(subject_dir) if f.endswith(".edf")])

        print(f"── {subject}: {len(edf_files)} EDF files ──────────────────")

        subject_epochs = []
        subject_labels = []

        for fname in edf_files:
            edf_path      = os.path.join(subject_dir, fname)
            seizure_times = seizure_map.get(fname, [])

            epochs, labels = process_file(edf_path, seizure_times)

            if epochs is not None:
                subject_epochs.append(epochs)
                subject_labels.append(labels)

        if subject_epochs:
            s_epochs = np.concatenate(subject_epochs, axis=0)
            s_labels = np.concatenate(subject_labels, axis=0)

            print(f"  {subject} summary:")
            print(f"    Interictal : {np.sum(s_labels == 0)}")
            print(f"    Preictal   : {np.sum(s_labels == 1)}")
            print(f"    Ictal      : {np.sum(s_labels == 2)}")
            print(f"    Total      : {len(s_labels)}\n")

            all_epochs.append(s_epochs)
            all_labels.append(s_labels)

    # Combine all subjects
    all_epochs = np.concatenate(all_epochs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"── Combined dataset summary ──────────────────────────")
    print(f"  Total epochs  : {len(all_labels)}")
    print(f"  Interictal (0): {np.sum(all_labels == 0)}")
    print(f"  Preictal   (1): {np.sum(all_labels == 1)}")
    print(f"  Ictal      (2): {np.sum(all_labels == 2)}")
    print(f"  Epoch shape   : {all_epochs.shape}")

    np.save(os.path.join(SAVE_DIR, "epochs.npy"), all_epochs)
    np.save(os.path.join(SAVE_DIR, "labels.npy"), all_labels)

    print(f"\n  Saved to {SAVE_DIR}")
    print("  Done!")


if __name__ == "__main__":
    main()