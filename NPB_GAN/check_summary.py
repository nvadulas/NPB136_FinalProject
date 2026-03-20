import os

BASE_DIR = os.path.expanduser(
    "~/Desktop/NPB_GAN/data/raw/physionet.org/files/chbmit/1.0.0"
)

# Check a subject that worked vs ones that didn't
for subject in ["chb01", "chb06", "chb10", "chb12"]:
    summary_path = os.path.join(BASE_DIR, subject, f"{subject}-summary.txt")
    if not os.path.exists(summary_path):
        print(f"{subject}: no summary file\n")
        continue

    print(f"── {subject} summary (first 40 lines) ───────────────")
    with open(summary_path) as f:
        for i, line in enumerate(f):
            if i > 40:
                break
            print(f"  {repr(line.rstrip())}")
    print()