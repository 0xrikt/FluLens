#!/usr/bin/env python3
"""
Prepare a reproducible influenza dataset from IDR idr0128.
Downloads selected plates, builds low/high infection labels, and creates
224x224 PNGs plus a manifest CSV.
"""
import argparse
import csv
import random
from ftplib import FTP
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HOST = "ftp.ebi.ac.uk"
BASE = "/pub/databases/IDR/idr0128-georgi-influenza/Influenza-20220113-Globus/IAV/3-Screen"

RAW_DIR = Path("/Users/rik/projects/FluLens/data/raw/idr0128")
PROC_DIR = Path("/Users/rik/projects/FluLens/data/processed/idr0128")
RAW_IMAGES = RAW_DIR / "images"
RAW_RESULTS = RAW_DIR / "results"
PROC_IMAGES = PROC_DIR / "images"

MANIFEST_PATH = PROC_DIR / "manifest.csv"

SEED = 42
LOW_Q = 0.1
HIGH_Q = 0.9

# Default caps to keep dataset manageable (Scheme A)
DEFAULT_MAX_TRAIN = 600
DEFAULT_MAX_VAL = 200
DEFAULT_MAX_TEST = 200

IMG_SIZE = (224, 224)


def ftp_connect():
    ftp = FTP(HOST)
    ftp.login()
    return ftp


def ftp_download(ftp: FTP, remote_path: str, local_path: Path):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        remote_size = ftp.size(remote_path)
    except Exception:
        remote_size = None
    if local_path.exists():
        if remote_size is None or local_path.stat().st_size == remote_size:
            return
    with open(local_path, "wb") as f:
        ftp.retrbinary("RETR " + remote_path, f.write)


def download_imagedata_csv(ftp: FTP, plate: str):
    filename = f"{plate}_ImageData.csv"
    remote = f"{BASE}/Data_UZH/Results/{filename}"
    local = RAW_RESULTS / filename
    ftp_download(ftp, remote, local)
    return local


def load_imagedata(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_to_uint8(img: Image.Image) -> Image.Image:
    # Convert to numpy, scale to 0-255
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        # If multi-channel, take first channel
        arr = arr[..., 0]
    minv, maxv = float(arr.min()), float(arr.max())
    if maxv <= minv:
        scaled = np.zeros_like(arr, dtype=np.uint8)
    else:
        scaled = ((arr - minv) / (maxv - minv) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def compose_rgb(nuc_path: Path, virus_path: Path, out_path: Path, ftp: FTP, nuc_remote: str, vir_remote: str):
    # Load images, normalize to 8-bit, resize, and compose into 3-channel
    try:
        nuc_img = Image.open(nuc_path)
    except Exception:
        ftp_download(ftp, nuc_remote, nuc_path)
        nuc_img = Image.open(nuc_path)
    try:
        vir_img = Image.open(virus_path)
    except Exception:
        ftp_download(ftp, vir_remote, virus_path)
        vir_img = Image.open(virus_path)
    nuc = normalize_to_uint8(nuc_img).resize(IMG_SIZE, Image.BILINEAR)
    vir = normalize_to_uint8(vir_img).resize(IMG_SIZE, Image.BILINEAR)
    rgb = Image.merge("RGB", (nuc, vir, vir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path, format="PNG")


def select_rows(df: pd.DataFrame, low_th: float, high_th: float) -> pd.DataFrame:
    low = df[df["numberOfInfectedNuclei"] <= low_th].copy()
    low["label"] = 0
    high = df[df["numberOfInfectedNuclei"] >= high_th].copy()
    high["label"] = 1
    return pd.concat([low, high], ignore_index=True)


def list_plates(ftp: FTP):
    plates_dir = f"{BASE}/Data_UZH/Screen"
    ftp.cwd(plates_dir)
    entries = []
    ftp.retrlines("LIST", entries.append)
    plates = []
    for line in entries:
        parts = line.split()
        name = parts[-1]
        if "Plate" in name:
            plates.append(name)
    return sorted(plates)


def split_plates(plates):
    # Deterministic split by plate order (Scheme A: 4 train, 1 val, 1 test)
    plates = plates[:6]
    train = plates[:4]
    val = plates[4:5]
    test = plates[5:6]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=DEFAULT_MAX_TRAIN)
    parser.add_argument("--max-val", type=int, default=DEFAULT_MAX_VAL)
    parser.add_argument("--max-test", type=int, default=DEFAULT_MAX_TEST)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    RAW_RESULTS.mkdir(parents=True, exist_ok=True)
    RAW_IMAGES.mkdir(parents=True, exist_ok=True)
    PROC_IMAGES.mkdir(parents=True, exist_ok=True)

    ftp = ftp_connect()

    plates = list_plates(ftp)
    train_plates, val_plates, test_plates = split_plates(plates)

    # Compute thresholds from training plates only
    train_frames = []
    for plate in train_plates:
        train_csv = download_imagedata_csv(ftp, plate)
        train_frames.append(load_imagedata(train_csv))
    train_df = pd.concat(train_frames, ignore_index=True)
    low_th = float(train_df["numberOfInfectedNuclei"].quantile(LOW_Q))
    high_th = float(train_df["numberOfInfectedNuclei"].quantile(HIGH_Q))

    all_rows = []

    for plate in plates[:6]:
        if plate in train_plates:
            split = "train"
        elif plate in val_plates:
            split = "val"
        else:
            split = "test"

        df = load_imagedata(download_imagedata_csv(ftp, plate))
        selected = select_rows(df, low_th, high_th)
        selected = selected.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        if split == "train":
            selected = selected.head(args.max_train)
        elif split == "val":
            selected = selected.head(args.max_val)
        else:
            selected = selected.head(args.max_test)

        plate_remote_dir = f"{BASE}/Data_UZH/Screen/{plate}"
        plate_raw_dir = RAW_IMAGES / plate
        plate_proc_dir = PROC_IMAGES / plate

        for _, row in selected.iterrows():
            nuc_name = row["NucleiImageName"]
            vir_name = row["VirusImageName"]

            nuc_remote = f"{plate_remote_dir}/{nuc_name}"
            vir_remote = f"{plate_remote_dir}/{vir_name}"

            nuc_local = plate_raw_dir / nuc_name
            vir_local = plate_raw_dir / vir_name

            ftp_download(ftp, nuc_remote, nuc_local)
            ftp_download(ftp, vir_remote, vir_local)

            out_name = nuc_name.replace("_w1.TIF", "_rgb.png")
            out_local = plate_proc_dir / out_name
            if not out_local.exists():
                compose_rgb(nuc_local, vir_local, out_local, ftp, nuc_remote, vir_remote)

            all_rows.append({
                "split": split,
                "plate": plate,
                "wellRow": row["wellRow"],
                "wellColumn": row["wellCollumn"],
                "label": int(row["label"]),
                "infected_nuclei": int(row["numberOfInfectedNuclei"]),
                "image_path": str(out_local),
            })

            # Remove raw files to save disk
            try:
                nuc_local.unlink(missing_ok=True)
                vir_local.unlink(missing_ok=True)
            except Exception:
                pass

    ftp.quit()

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "plate", "wellRow", "wellColumn", "label", "infected_nuclei", "image_path"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved manifest to {MANIFEST_PATH}")
    print(f"Low threshold: {low_th:.1f}, High threshold: {high_th:.1f}")
    print(f"Total samples: {len(all_rows)}")


if __name__ == "__main__":
    main()
