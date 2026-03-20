import os
import subprocess

# Your PhysioNet username
PHYSIONET_USER = "nvadulas"   # ← change this
PHYSIONET_PASS = "Mul@nak3!3V@dulas1"

BASE_URL  = "https://physionet.org/files/chbmit/1.0.0"
BASE_DIR  = os.path.expanduser("~/Desktop/NPB_GAN/data/raw/physionet.org/files/chbmit/1.0.0")

# All seizure-containing files from your list
SEIZURE_FILES = [
    "chb02/chb02_16.edf",
    "chb02/chb02_19.edf",
    "chb03/chb03_01.edf",
    "chb03/chb03_02.edf",
    "chb03/chb03_03.edf",
    "chb03/chb03_04.edf",
    "chb03/chb03_34.edf",
    "chb03/chb03_35.edf",
    "chb03/chb03_36.edf",
    "chb04/chb04_05.edf",
    "chb04/chb04_08.edf",
    "chb04/chb04_28.edf",
    "chb05/chb05_06.edf",
    "chb05/chb05_13.edf",
    "chb05/chb05_16.edf",
    "chb05/chb05_17.edf",
    "chb05/chb05_22.edf",
    "chb06/chb06_01.edf",
    "chb06/chb06_04.edf",
    "chb06/chb06_09.edf",
    "chb06/chb06_10.edf",
    "chb06/chb06_13.edf",
    "chb06/chb06_18.edf",
    "chb06/chb06_24.edf",
    "chb07/chb07_12.edf",
    "chb07/chb07_13.edf",
    "chb07/chb07_18.edf",
    "chb08/chb08_02.edf",
    "chb08/chb08_05.edf",
    "chb08/chb08_11.edf",
    "chb08/chb08_13.edf",
    "chb08/chb08_21.edf",
    "chb09/chb09_06.edf",
    "chb09/chb09_08.edf",
    "chb09/chb09_19.edf",
    "chb10/chb10_12.edf",
    "chb10/chb10_20.edf",
    "chb10/chb10_27.edf",
    "chb10/chb10_30.edf",
    "chb10/chb10_31.edf",
    "chb10/chb10_38.edf",
    "chb11/chb11_82.edf",
    "chb11/chb11_92.edf",
    "chb11/chb11_99.edf",
    "chb12/chb12_06.edf",
    "chb12/chb12_08.edf",
    "chb12/chb12_09.edf",
    "chb12/chb12_10.edf",
    "chb12/chb12_11.edf",
    "chb12/chb12_23.edf",
    "chb12/chb12_27.edf",
    "chb12/chb12_28.edf",
    "chb12/chb12_29.edf",
    "chb12/chb12_33.edf",
    "chb12/chb12_36.edf",
    "chb12/chb12_38.edf",
    "chb12/chb12_42.edf",
    "chb13/chb13_19.edf",
    "chb13/chb13_21.edf",
    "chb13/chb13_40.edf",
    "chb13/chb13_55.edf",
    "chb13/chb13_58.edf",
]

# Get unique subject folders so we can also grab summary files
SUBJECTS = sorted(set(f.split("/")[0] for f in SEIZURE_FILES))


def download_file(url, dest_path):
    """Download a single file using wget, skip if already exists."""
    if os.path.exists(dest_path):
        print(f"  Already exists, skipping: {os.path.basename(dest_path)}")
        return

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    cmd = [
    "wget", "-c", "-q",
    f"--user={PHYSIONET_USER}",
    f"--password={PHYSIONET_PASS}",
    "-O", dest_path,
    url
]

    print(f"  Downloading: {os.path.basename(dest_path)} ...")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"  ✗ Failed: {dest_path}")
    else:
        print(f"  ✓ Done: {os.path.basename(dest_path)}")


def main():
    print(f"── Downloading seizure EDF files ─────────────────────")
    print(f"   {len(SEIZURE_FILES)} EDF files across {len(SUBJECTS)} subjects\n")

    # Download each seizure EDF file
    for rel_path in SEIZURE_FILES:
        url       = f"{BASE_URL}/{rel_path}"
        dest_path = os.path.join(BASE_DIR, rel_path)
        download_file(url, dest_path)

    # Download summary files for each subject (needed for seizure timestamps)
    print(f"\n── Downloading summary files ──────────────────────────")
    for subject in SUBJECTS:
        summary_file = f"{subject}/{subject}-summary.txt"
        url          = f"{BASE_URL}/{summary_file}"
        dest_path    = os.path.join(BASE_DIR, summary_file)
        download_file(url, dest_path)

    print(f"\n── Download complete ──────────────────────────────────")
    print(f"   Check files at: {BASE_DIR}")


if __name__ == "__main__":
    main()