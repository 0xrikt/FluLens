#!/usr/bin/env python3
import argparse
from ftplib import FTP
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

HOST = "ftp.ebi.ac.uk"
BASE = "/pub/databases/IDR/idr0128-georgi-influenza/Influenza-20220113-Globus/IAV/3-Screen"
RAW_RESULTS = Path("/Users/rik/projects/FluLens/data/raw/idr0128/results")
RAW_IMAGES = Path("/Users/rik/projects/FluLens/data/raw/idr0128/images")
PROC_IMAGES = Path("/Users/rik/projects/FluLens/data/processed/idr0128/images")

IMG_SIZE = (224, 224)

PLATES = [
    "180426-6-53-IAV-1A_Plate_11962",
    "180426-6-53-IAV-1B_Plate_11961",
    "180426-6-53-IAV-1C_Plate_11963",
    "180426-6-53-IAV-1D_Plate_11964",
    "180513-6-54-IAV-2A_Plate_12012",
    "180513-6-54-IAV-2B_Plate_12011",
]


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


def normalize_to_uint8(img: Image.Image) -> Image.Image:
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    minv, maxv = float(arr.min()), float(arr.max())
    if maxv <= minv:
        scaled = np.zeros_like(arr, dtype=np.uint8)
    else:
        scaled = ((arr - minv) / (maxv - minv) * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def compose_rgb(nuc_path: Path, virus_path: Path, out_path: Path):
    nuc = normalize_to_uint8(Image.open(nuc_path)).resize(IMG_SIZE, Image.BILINEAR)
    vir = normalize_to_uint8(Image.open(virus_path)).resize(IMG_SIZE, Image.BILINEAR)
    rgb = Image.merge("RGB", (nuc, vir, vir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb.save(out_path, format="PNG")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-plate", type=int, default=30)
    args = parser.parse_args()

    # thresholds from first 4 plates
    train_frames = [pd.read_csv(RAW_RESULTS / f"{p}_ImageData.csv") for p in PLATES[:4]]
    train_df = pd.concat(train_frames, ignore_index=True)
    low = float(train_df["numberOfInfectedNuclei"].quantile(0.1))
    high = float(train_df["numberOfInfectedNuclei"].quantile(0.9))

    ftp = ftp_connect()

    for plate in PLATES:
        df = pd.read_csv(RAW_RESULTS / f"{plate}_ImageData.csv")
        low_df = df[df["numberOfInfectedNuclei"] <= low].head(args.per_plate // 2)
        high_df = df[df["numberOfInfectedNuclei"] >= high].head(args.per_plate // 2)
        selected = pd.concat([low_df, high_df], ignore_index=True)
        plate_remote = f"{BASE}/Data_UZH/Screen/{plate}"
        plate_raw_dir = RAW_IMAGES / plate
        plate_proc_dir = PROC_IMAGES / plate

        for _, row in selected.iterrows():
            nuc_name = row["NucleiImageName"]
            vir_name = row["VirusImageName"]

            out_name = nuc_name.replace("_w1.TIF", "_rgb.png")
            out_path = plate_proc_dir / out_name
            if out_path.exists():
                continue

            for name in [nuc_name, vir_name]:
                remote = f"{plate_remote}/{name}"
                local = plate_raw_dir / name
                ftp_download(ftp, remote, local)

            try:
                compose_rgb(plate_raw_dir / nuc_name, plate_raw_dir / vir_name, out_path)
            except Exception:
                continue

            # cleanup raw to save disk
            try:
                (plate_raw_dir / nuc_name).unlink(missing_ok=True)
                (plate_raw_dir / vir_name).unlink(missing_ok=True)
            except Exception:
                pass

        print("seeded", plate)

    ftp.quit()


if __name__ == "__main__":
    main()
