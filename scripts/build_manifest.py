#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path

import pandas as pd

SEED = 42
LOW_Q = 0.1
HIGH_Q = 0.9

RAW_RESULTS = Path("/Users/rik/projects/FluLens/data/raw/idr0128/results")
PROC_IMAGES = Path("/Users/rik/projects/FluLens/data/processed/idr0128/images")
MANIFEST_PATH = Path("/Users/rik/projects/FluLens/data/processed/idr0128/manifest.csv")


def select_rows(df: pd.DataFrame, low_th: float, high_th: float):
    low = df[df["numberOfInfectedNuclei"] <= low_th].copy()
    low["label"] = 0
    high = df[df["numberOfInfectedNuclei"] >= high_th].copy()
    high["label"] = 1
    return pd.concat([low, high], ignore_index=True)


def list_plates():
    plates = []
    for p in PROC_IMAGES.glob("*_Plate_*"):
        if p.is_dir():
            plates.append(p.name)
    return sorted(plates)


def split_plates(plates):
    plates = plates[:6]
    train = plates[:4]
    val = plates[4:5]
    test = plates[5:6]
    return train, val, test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=600)
    parser.add_argument("--max-val", type=int, default=200)
    parser.add_argument("--max-test", type=int, default=200)
    args = parser.parse_args()

    random.seed(SEED)

    plates = list_plates()
    train_plates, val_plates, test_plates = split_plates(plates)

    train_frames = []
    for plate in train_plates:
        train_csv = RAW_RESULTS / f"{plate}_ImageData.csv"
        if train_csv.exists():
            train_frames.append(pd.read_csv(train_csv))
    train_df = pd.concat(train_frames, ignore_index=True)
    low_th = float(train_df["numberOfInfectedNuclei"].quantile(LOW_Q))
    high_th = float(train_df["numberOfInfectedNuclei"].quantile(HIGH_Q))

    all_rows = []

    for plate in plates:
        if plate in train_plates:
            split = "train"
        elif plate in val_plates:
            split = "val"
        else:
            split = "test"

        csv_path = RAW_RESULTS / f"{plate}_ImageData.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        selected = select_rows(df, low_th, high_th)
        selected = selected.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        if split == "train":
            selected = selected.head(args.max_train)
        elif split == "val":
            selected = selected.head(args.max_val)
        else:
            selected = selected.head(args.max_test)

        for _, row in selected.iterrows():
            nuc_name = row["NucleiImageName"]
            out_name = nuc_name.replace("_w1.TIF", "_rgb.png")
            out_local = PROC_IMAGES / plate / out_name
            if not out_local.exists():
                continue
            all_rows.append({
                "split": split,
                "plate": plate,
                "wellRow": row["wellRow"],
                "wellColumn": row["wellCollumn"],
                "label": int(row["label"]),
                "infected_nuclei": int(row["numberOfInfectedNuclei"]),
                "image_path": str(out_local),
            })

    train = [r for r in all_rows if r["split"] == "train"][: args.max_train]
    val = [r for r in all_rows if r["split"] == "val"][: args.max_val]
    test = [r for r in all_rows if r["split"] == "test"][: args.max_test]

    def balance(rows):
        zeros = [r for r in rows if r["label"] == 0]
        ones = [r for r in rows if r["label"] == 1]
        n = min(len(zeros), len(ones))
        return zeros[:n] + ones[:n]

    train = balance(train)
    val = balance(val)
    test = balance(test)
    all_rows = train + val + test

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
