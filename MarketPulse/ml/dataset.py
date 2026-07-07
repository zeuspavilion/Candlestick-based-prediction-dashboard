import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from MarketPulse.config import (
    IMAGE_SIZE,
    IMAGE_DPI,
    LABELS,
    SPLIT_RATIOS,
    PURGE_SAMPLES_BETWEEN_SPLITS,
    CLASS_TO_IDX,
    PROJECT_ROOT,
    IMAGE_DIR,
    METADATA_PATH,
    CLASS_MAPPING_PATH,
    MANIFEST_DIR,
    SUMMARY_PATH,
    CANDLES_PER_DAY,
    LOOKBACK_DAYS,
    WINDOW_CANDLES,
    REQUIRED_COLUMNS,
)

# Resampling filter safety check (handles different PIL versions)
RESAMPLE_FILTER = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

class CandlestickManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, transform=None):
        self.manifest_path = Path(manifest_path)
        self.samples = pd.read_csv(self.manifest_path)
        self.transform = transform

        if self.samples.empty:
            raise ValueError(f"Manifest is empty: {self.manifest_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        image_path = PROJECT_ROOT / row["image_path"]
        image = Image.open(image_path).convert("RGB")
        label = int(row["label_id"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def save_candlestick_image(window_df: pd.DataFrame, output_path: Path):
    plot_df = window_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    plot_df.index = pd.to_datetime(plot_df.index)

    style = mpf.make_mpf_style(
        base_mpf_style="yahoo",
        marketcolors=mpf.make_marketcolors(
            up="#1a9850", down="#d73027", edge="inherit", wick="inherit", volume="inherit"
        ),
        facecolor="white",
        figcolor="white",
    )

    fig, _ = mpf.plot(
        plot_df,
        type="candle",
        style=style,
        volume=False,
        axisoff=True,
        returnfig=True,
        figratio=(1, 1),
        figscale=1.0,
        tight_layout=True,
        warn_too_much_data=1000,
    )

    fig.set_size_inches(IMAGE_SIZE[0] / IMAGE_DPI, IMAGE_SIZE[1] / IMAGE_DPI)
    fig.savefig(output_path, dpi=IMAGE_DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # Enforce exact size and RGB format
    with Image.open(output_path) as img:
        img = img.convert("RGB").resize(IMAGE_SIZE, RESAMPLE_FILTER)
        img.save(output_path, format="PNG", optimize=True)


def assign_purged_chronological_splits(records: list) -> tuple:
    split_records = []
    dropped_for_purge = []

    for ticker in sorted({record["ticker"] for record in records}):
        ticker_records = sorted(
            [record for record in records if record["ticker"] == ticker],
            key=lambda r: r["end_date"],
        )
        n = len(ticker_records)
        train_end = int(n * SPLIT_RATIOS["train"])
        validation_end = train_end + int(n * SPLIT_RATIOS["validation"])

        validation_start = train_end + PURGE_SAMPLES_BETWEEN_SPLITS
        test_start = validation_end + PURGE_SAMPLES_BETWEEN_SPLITS

        for idx, record in enumerate(ticker_records):
            if idx < train_end:
                split = "train"
            elif idx < validation_start:
                dropped_for_purge.append({**record, "purge_reason": "train_validation_boundary"})
                continue
            elif idx < validation_end:
                split = "validation"
            elif idx < test_start:
                dropped_for_purge.append({**record, "purge_reason": "validation_test_boundary"})
                continue
            else:
                split = "test"

            updated = record.copy()
            updated["split"] = split
            split_records.append(updated)

    return split_records, dropped_for_purge


def audit_cross_split_input_overlap(records: list) -> list:
    issues = []
    for ticker in sorted({record["ticker"] for record in records}):
        ticker_records = [record for record in records if record["ticker"] == ticker]
        dates_by_split = {split: set() for split in SPLIT_RATIOS}

        for record in ticker_records:
            context_dates = set(record["context_dates"].split("|"))
            dates_by_split[record["split"]].update(context_dates)

        split_pairs = [("train", "validation"), ("validation", "test"), ("train", "test")]
        for left, right in split_pairs:
            overlap = dates_by_split[left] & dates_by_split[right]
            if overlap:
                issues.append({
                    "ticker": ticker,
                    "left_split": left,
                    "right_split": right,
                    "overlap_count": len(overlap),
                })
    return issues


def generate_image_dataset(records: list) -> pd.DataFrame:
    # Ensure directories
    for split in SPLIT_RATIOS:
        for label in LABELS:
            (IMAGE_DIR / split / label).mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    for record in tqdm(records, desc="Generating candlestick images"):
        output_path = IMAGE_DIR / record["split"] / record["label"] / f"{record['image_id']}.png"
        
        # Save image if it doesn't exist
        if not output_path.exists():
            save_candlestick_image(record["window_df"], output_path)

        metadata_row = {key: value for key, value in record.items() if key != "window_df"}
        metadata_row["image_path"] = str(output_path.relative_to(PROJECT_ROOT))
        metadata_row["image_width"] = IMAGE_SIZE[0]
        metadata_row["image_height"] = IMAGE_SIZE[1]
        metadata_row["candles_per_image"] = WINDOW_CANDLES
        metadata_rows.append(metadata_row)

    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(METADATA_PATH, index=False)
    return metadata


def build_and_verify_dataset(all_records: list):
    """Orchestrates split, purge, image generation, and manifest creation."""
    # 1. Split & Purge
    split_records, purged = assign_purged_chronological_splits(all_records)
    
    # 2. Check overlap leakage
    leakage_issues = audit_cross_split_input_overlap(split_records)
    if leakage_issues:
        raise ValueError(f"Data leakage detected during chronological split: {leakage_issues}")

    # 3. Generate Image Dataset
    metadata = generate_image_dataset(split_records)

    # 4. Save Class Mapping
    with open(CLASS_MAPPING_PATH, "w") as f:
        json.dump(CLASS_TO_IDX, f, indent=2)

    # 5. Create manifests per split
    manifest_cols = [
        "image_path", "label", "label_id", "split", "ticker", "stock_name",
        "category", "start_date", "end_date", "context_dates", "label_date",
        "current_close", "next_close", "next_day_return", "rolling_20d_volatility",
        "dynamic_threshold", "candles_per_image"
    ]

    for split in SPLIT_RATIOS:
        split_df = metadata[metadata["split"] == split].copy()
        split_df["label_id"] = split_df["label"].map(CLASS_TO_IDX).astype(int)
        split_df = split_df.sort_values(["ticker", "end_date", "label_date"]).reset_index(drop=True)
        
        split_manifest_path = MANIFEST_DIR / f"{split}.csv"
        split_df[manifest_cols].to_csv(split_manifest_path, index=False)

    all_splits_path = MANIFEST_DIR / "all_splits.csv"
    metadata["label_id"] = metadata["label"].map(CLASS_TO_IDX).astype(int)
    metadata[manifest_cols].sort_values(["split", "ticker", "end_date"]).to_csv(all_splits_path, index=False)

    # 6. Save summary json
    summary = {
        "total_images": int(len(metadata)),
        "image_size": IMAGE_SIZE,
        "labels": LABELS,
        "purged_records": int(len(purged)),
        "split_counts": metadata["split"].value_counts().to_dict(),
        "label_counts": metadata["label"].value_counts().to_dict(),
        "class_to_idx": CLASS_TO_IDX,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Dataset built. Manifests saved under: {MANIFEST_DIR}")
    return metadata
