#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path

import pandas as pd

SEED = 42
LOW_Q = 0.1
HIGH_Q = 0.9

TRAIN_PLATE = "180426-6-53-IAV-1A_Plate_11962"
TEST_PLATE = "180426-6-53-IAV-1B_Plate_11961"

RAW_RESULTS = Path("/Users/rik/projects/FluLens/data/raw/idr0128/results")
PROC_IMAGES = Path("/Users/rik/projects/FluLens/data/processed/idr0128/images")
MANIFEST_PATH = Path("/Users/rik/projects/FluLens/data/processed/idr0128/manifest.csv")


def select_rows(df: pd.DataFrame, low_th: float, high_th: float):
    low = df[df["numberOfInfectedNuclei"] <= low_th].copy()
    low["label"] = 0
    high = df[df["numberOfInfectedNuclei"] >= high_th].copy()
    high["label"] = 1
    return pd.concat([low, high], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=40)
    parser.add_argument("--max-val", type=int, default=10)
    parser.add_argument("--max-test", type=int, default=40)
    args = parser.parse_args()

    random.seed(SEED)

    train_csv = RAW_RESULTS / f"{TRAIN_PLATE}_ImageData.csv"
    test_csv = RAW_RESULTS / f"{TEST_PLATE}_ImageData.csv"

    train_df = pd.read_csv(train_csv)
    low_th = float(train_df["numberOfInfectedNuclei"].quantile(LOW_Q))
    high_th = float(train_df["numberOfInfectedNuclei"].quantile(HIGH_Q))

    all_rows = []

    for plate, split in [(TRAIN_PLATE, "trainval"), (TEST_PLATE, "test")]:
        df = pd.read_csv(RAW_RESULTS / f"{plate}_ImageData.csv")
        selected = select_rows(df, low_th, high_th)
        selected = selected.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        if split == "trainval":
            selected = selected.head(args.max_train + args.max_val)
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

    trainval = [r for r in all_rows if r["split"] == "trainval"]
    random.shuffle(trainval)
    val_size = max(1, int(len(trainval) * 0.25))
    val_set = set(id(x) for x in trainval[:val_size])

    for r in all_rows:
        if r["split"] == "trainval":
            r["split"] = "val" if id(r) in val_set else "train"

    train = [r for r in all_rows if r["split"] == "train"][: args.max_train]
    val = [r for r in all_rows if r["split"] == "val"][: args.max_val]
    test = [r for r in all_rows if r["split"] == "test"][: args.max_test]
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
