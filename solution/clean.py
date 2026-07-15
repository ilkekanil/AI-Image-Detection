#!/usr/bin/env python3
"""
This script pulls raw training data from `data/train/*.parquet` and writes everything 
clean and tidy into `artifacts/task01/`. Run it directly from your solution folder like so:

    python clean.py --timeout_seconds 600

To keep things consistent and predictable, the cleaning step is entirely deterministic. 
We'll convert every readable image to RGB and resize it to a uniform 224x224, holding off 
on any random data augmentations for now.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path

# Matplotlib defaults to writing caches to the user's home directory, which often breaks 
# inside Docker or read-only environments. Pointing it to a temp folder keeps everything 
# self-contained and headache-free.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "amls_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError


CLEANED_SIZE = 224
JPEG_QUALITY = 90
STALE_ROOT_FILES = [
    "metadata.csv",
    "binary_distribution.png",
    "byte_length_by_class.png",
    "class_distribution.png",
    "image_size_scatter.png",
    "top_dimensions.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explore and clean the AMLS training split.")
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.add_argument("--data_dir", type=Path, default=None)
    parser.add_argument("--artifacts_dir", type=Path, default=None)
    return parser.parse_args()


def resolve_data_dir(data_dir_arg: Path | None) -> Path:
    """Locates the training data directory while leaving other splits untouched.

    During autograding, the script expects to find the data at `solution/data/train`. 
    When working locally in this workspace, the files might also live one level up.
    """
    if data_dir_arg is not None:
        data_dir = data_dir_arg
        if not (data_dir / "train").is_dir():
            raise FileNotFoundError(f"Could not find train/ inside {data_dir}")
        return data_dir

    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    candidates = [
        script_dir / "data",
        cwd / "data",
        script_dir.parent,
        cwd,
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir():
            return candidate

    raise FileNotFoundError("Could not find data/train/*.parquet")


def resolve_artifacts_dir(artifacts_dir_arg: Path | None) -> Path:
    if artifacts_dir_arg is not None:
        return artifacts_dir_arg
    return Path(__file__).resolve().parent / "artifacts"


def binary_label(source_class: int) -> int:
    return 0 if int(source_class) == 0 else 1


def class_name(source_class: int) -> str:
    names = {
        0: "real",
        1: "sd_2_1",
        2: "sdxl",
        3: "sd_3",
        4: "dall_e_3",
        5: "midjourney",
    }
    return names.get(int(source_class), f"class_{source_class}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def log(message: str) -> None:
    print(message, flush=True)


def remove_file_if_exists(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def clean_previous_task01_outputs(task_dir: Path) -> None:
    """Clears out any leftover files from previous runs of this script so we start fresh."""
    for filename in STALE_ROOT_FILES:
        remove_file_if_exists(task_dir / filename)

    cleaned_dir = task_dir / "cleaned_train"
    for parquet_path in cleaned_dir.glob("cleaned_train_*.parquet"):
        parquet_path.unlink()

    images_dir = cleaned_dir / "images"
    for image_path in images_dir.glob("*.jpg"):
        image_path.unlink()


def parquet_files(train_dir: Path) -> list[Path]:
    files = sorted(train_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {train_dir}")
    return files


def numeric_summary(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "mean": float(series.mean()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
    }


def decode_rgb(image_bytes: bytes) -> Image.Image:

    with Image.open(BytesIO(image_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def image_statistics(image: Image.Image, byte_size: int) -> dict[str, float | int]:
    width, height = image.size
    arr = np.asarray(image, dtype=np.float32)
    gray = arr.mean(axis=2)
    rgb_means = arr.reshape(-1, 3).mean(axis=0)
    rgb_stds = arr.reshape(-1, 3).std(axis=0)

    return {
        "width": int(width),
        "height": int(height),
        "aspect_ratio": float(width / height) if height else math.nan,
        "byte_size": int(byte_size),
        "mean_brightness": float(gray.mean()),
        "pixel_std": float(gray.std()),
        "mean_r": float(rgb_means[0]),
        "mean_g": float(rgb_means[1]),
        "mean_b": float(rgb_means[2]),
        "std_r": float(rgb_stds[0]),
        "std_g": float(rgb_stds[1]),
        "std_b": float(rgb_stds[2]),
    }


def save_cleaned_image(image: Image.Image, output_path: Path) -> None:
    """Saves the cleaned image using a deterministic process.

    Resizing to 224x224 stops the model from 'cheating' by using original image dimensions, 
    and it keeps downstream training light enough for local CPU runs. We are skipping 
    data augmentations (like crops or jitter) here because this stage is strictly about 
    data exploration and preparation.
    """
    cleaned = image.resize((CLEANED_SIZE, CLEANED_SIZE), Image.Resampling.BICUBIC)
    cleaned.save(output_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)


def plot_bar(labels: list[str], values: list[int], title: str, ylabel: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color="#4c78a8")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_hist(values: pd.Series, title: str, xlabel: str, output_path: Path, bins: int = 40) -> None:
    plt.figure(figsize=(7.5, 4.5))
    plt.hist(values.dropna(), bins=bins, color="#59a14f", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("images")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_box_by_class(
    metadata: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    source_classes = sorted(metadata["source_class"].unique())
    data = [metadata.loc[metadata["source_class"] == cls, column].values for cls in source_classes]
    labels = [f"{cls}: {class_name(cls)}" for cls in source_classes]

    plt.figure(figsize=(8, 4.8))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def create_plots(metadata: pd.DataFrame, plots_dir: Path) -> None:
    class_counts = metadata["source_class"].value_counts().sort_index()
    plot_bar(
        labels=[f"{idx}: {class_name(idx)}" for idx in class_counts.index],
        values=[int(value) for value in class_counts.values],
        title="Class distribution by original source_class",
        ylabel="images",
        output_path=plots_dir / "class_distribution_source_class.png",
    )

    binary_counts = metadata["binary_label"].value_counts().sort_index()
    plot_bar(
        labels=["real (0)", "AI-generated (1)"],
        values=[int(binary_counts.get(0, 0)), int(binary_counts.get(1, 0))],
        title="Binary class distribution",
        ylabel="images",
        output_path=plots_dir / "binary_class_distribution.png",
    )

    plot_hist(metadata["width"], "Image width distribution", "width", plots_dir / "width_distribution.png")
    plot_hist(metadata["height"], "Image height distribution", "height", plots_dir / "height_distribution.png")
    plot_hist(
        metadata["aspect_ratio"],
        "Aspect ratio distribution",
        "width / height",
        plots_dir / "aspect_ratio_distribution.png",
    )
    plot_hist(
        metadata["byte_size"],
        "Image byte size distribution",
        "JPEG bytes",
        plots_dir / "byte_size_distribution.png",
    )
    plot_box_by_class(
        metadata,
        "mean_brightness",
        "Brightness distribution by source_class",
        "mean brightness",
        plots_dir / "brightness_by_source_class.png",
    )
    plot_box_by_class(
        metadata,
        "pixel_std",
        "Contrast distribution by source_class",
        "pixel standard deviation",
        plots_dir / "contrast_by_source_class.png",
    )


def summarize(metadata: pd.DataFrame, total_images: int, skipped: list[dict]) -> dict:
    metadata = metadata.copy()
    metadata["dimension"] = metadata["width"].astype(str) + "x" + metadata["height"].astype(str)

    class_distribution = {
        str(int(k)): int(v) for k, v in metadata["source_class"].value_counts().sort_index().items()
    }
    binary_distribution = {
        str(int(k)): int(v) for k, v in metadata["binary_label"].value_counts().sort_index().items()
    }

    per_class = {}
    for source_class, group in metadata.groupby("source_class"):
        per_class[str(int(source_class))] = {
            "class_name": class_name(int(source_class)),
            "count": int(len(group)),
            "width": numeric_summary(group["width"]),
            "height": numeric_summary(group["height"]),
            "aspect_ratio": numeric_summary(group["aspect_ratio"]),
            "byte_size": numeric_summary(group["byte_size"]),
            "mean_brightness": numeric_summary(group["mean_brightness"]),
            "pixel_std": numeric_summary(group["pixel_std"]),
            "top_dimensions": {
                str(k): int(v) for k, v in group["dimension"].value_counts().head(10).items()
            },
        }

    class_counts = metadata.groupby("source_class").size()
    images_320 = metadata[(metadata["width"] == 320) & (metadata["height"] == 320)]
    share_320 = images_320.groupby("source_class").size().reindex(class_counts.index, fill_value=0) / class_counts
    shortcut_findings = [
        f"source_class {int(cls)} ({class_name(int(cls))}) has {share:.1%} images with original size 320x320"
        for cls, share in share_320.sort_index().items()
    ]
    class_4_270 = metadata[
        (metadata["source_class"] == 4) & (metadata["width"] == 270) & (metadata["height"] == 270)
    ]
    if len(class_4_270):
        shortcut_findings.append(
            f"source_class 4 ({class_name(4)}) has {len(class_4_270)}/{int((metadata['source_class'] == 4).sum())} images with original size 270x270"
        )

    return {
        "total_images": int(total_images),
        "valid_images": int(len(metadata)),
        "corrupted_or_skipped_images": int(len(skipped)),
        "skipped_images": skipped,
        "class_distribution": class_distribution,
        "binary_distribution": binary_distribution,
        "overall": {
            "width": numeric_summary(metadata["width"]),
            "height": numeric_summary(metadata["height"]),
            "aspect_ratio": numeric_summary(metadata["aspect_ratio"]),
            "byte_size": numeric_summary(metadata["byte_size"]),
            "mean_brightness": numeric_summary(metadata["mean_brightness"]),
            "pixel_std": numeric_summary(metadata["pixel_std"]),
            "mean_r": numeric_summary(metadata["mean_r"]),
            "mean_g": numeric_summary(metadata["mean_g"]),
            "mean_b": numeric_summary(metadata["mean_b"]),
            "std_r": numeric_summary(metadata["std_r"]),
            "std_g": numeric_summary(metadata["std_g"]),
            "std_b": numeric_summary(metadata["std_b"]),
        },
        "per_source_class": per_class,
        "top_dimensions": {
            str(k): int(v) for k, v in metadata["dimension"].value_counts().head(20).items()
        },
        "potential_shortcut_findings": shortcut_findings,
        "cleaning": {
            "rgb_conversion": "All valid images are converted to RGB for a consistent three-channel input.",
            "resize": f"All valid images are resized deterministically to {CLEANED_SIZE}x{CLEANED_SIZE}.",
            "augmentation": "No augmentation is applied in Task 1.1.",
        },
    }


def write_cleaning_notes(path: Path, summary: dict) -> None:
    class_lines = "\n".join(
        f"- source_class {cls}: {count}" for cls, count in summary["class_distribution"].items()
    )
    content = f"""Task 1.1 cleaning notes

Total images found in data/train: {summary['total_images']}
Valid images: {summary['valid_images']}
Corrupted/skipped images: {summary['corrupted_or_skipped_images']}

Class distribution:
{class_lines}

RGB conversion:
All valid images are converted to RGB so that downstream models receive a consistent three-channel input.
This avoids special cases caused by grayscale, palette, or other image modes.

Resize to 224x224:
All valid images are resized to 224x224 to remove original image-size shortcuts and to keep later CPU
training efficient. The operation is deterministic and therefore reproducible.

No augmentation:
Task 1.1 is about exploration and cleaning, not robustness training. Random crops, color jitter, blur,
compression changes, and other augmentation methods are intentionally left for later tasks.
"""
    path.write_text(content, encoding="utf-8")


def write_task01_report(path: Path, summary: dict) -> None:
    class_rows = []
    for source_class, info in summary["per_source_class"].items():
        top_dimension, top_count = next(iter(info["top_dimensions"].items()))
        class_rows.append(
            f"| {source_class} | {info['class_name']} | {info['count']} | "
            f"{top_dimension} ({top_count}) | {info['byte_size']['mean']:.0f} | "
            f"{info['mean_brightness']['mean']:.2f} | {info['pixel_std']['mean']:.2f} |"
        )

    shortcut_lines = "\n".join(
        f"- {finding}" for finding in summary["potential_shortcut_findings"]
    )
    plot_lines = "\n".join(
        [
            "- `plots/class_distribution_source_class.png`",
            "- `plots/binary_class_distribution.png`",
            "- `plots/width_distribution.png`",
            "- `plots/height_distribution.png`",
            "- `plots/aspect_ratio_distribution.png`",
            "- `plots/byte_size_distribution.png`",
            "- `plots/brightness_by_source_class.png`",
            "- `plots/contrast_by_source_class.png`",
        ]
    )

    content = f"""# Task 1.1 Dataset Exploration and Cleaning

## Dataset Summary

The training split contains {summary['total_images']} images. {summary['valid_images']} images were decoded
successfully and {summary['corrupted_or_skipped_images']} images were skipped as corrupted or unreadable.
The original six source classes are preserved in the metadata. For later binary modeling, class `0` remains
`0` (real) and classes `1` to `5` are mapped to binary label `1` (AI-generated).

Binary distribution:

- real (`0`): {summary['binary_distribution'].get('0', 0)}
- AI-generated (`1`): {summary['binary_distribution'].get('1', 0)}

| source_class | meaning | images | most common original size | mean bytes | mean brightness | mean contrast |
|---:|---|---:|---|---:|---:|---:|
{chr(10).join(class_rows)}

## Characteristics and Shortcut Risks

The exploration computes original width, height, aspect ratio, JPEG byte size, brightness, contrast, RGB
channel means, and RGB channel standard deviations for every valid image. The most important shortcut risk
is that original image dimensions are strongly related to the source class:

{shortcut_lines}

Because later splits are standardized differently, a model should not be allowed to rely on original training
dimensions as a class cue. This motivates deterministic resizing during cleaning.

## Cleaning Pipeline

The cleaning pipeline is deterministic and applies exactly one cleaned output to each valid training image:

1. Decode JPEG bytes using Pillow.
2. Apply EXIF orientation handling.
3. Convert to RGB.
4. Resize to {CLEANED_SIZE}x{CLEANED_SIZE} with bicubic interpolation.
5. Save the cleaned result as a JPEG image.
6. Save `labels.csv` with image path, original source class, binary label, original size, and cleaned size.

RGB conversion is used to give later CPU-friendly models a consistent three-channel input. Resizing to
224x224 removes image-size shortcuts and reduces downstream training cost. No random augmentation is applied
because Task 1.1 is limited to exploration and cleaning; robustness augmentation belongs to later tasks.

## Current Output Files

- `train_metadata.csv`
- `summary.json`
- `cleaning_notes.txt`
- `cleaned_train/images/*.jpg`
- `cleaned_train/labels.csv`

Plots:

{plot_lines}
"""
    path.write_text(content, encoding="utf-8")


def process_train_split(
    train_dir: Path,
    task_dir: Path,
    deadline: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], int]:
    images_dir = ensure_dir(task_dir / "cleaned_train" / "images")
    metadata_rows: list[dict] = []
    label_rows: list[dict] = []
    skipped: list[dict] = []
    total_images = 0

    files = parquet_files(train_dir)
    log(f"Found {len(files)} parquet files in {train_dir}")

    for file_index, parquet_path in enumerate(files, start=1):
        log(f"[{file_index}/{len(files)}] Reading {parquet_path.name}")
        df = pd.read_parquet(parquet_path, columns=["image", "source_class"])

        for row_index, row in df.iterrows():
            if time.monotonic() > deadline:
                raise TimeoutError("Timeout reached before clean.py finished.")

            total_images += 1
            source_class = int(row["source_class"])
            label = binary_label(source_class)
            image_id = f"{parquet_path.stem}_{int(row_index):06d}"
            cleaned_name = f"{image_id}.jpg"
            cleaned_path = images_dir / cleaned_name

            try:
                image_bytes = row["image"]
                image = decode_rgb(image_bytes)
                stats = image_statistics(image, len(image_bytes))
                save_cleaned_image(image, cleaned_path)
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                skipped.append(
                    {
                        "source_file": parquet_path.name,
                        "source_row": int(row_index),
                        "source_class": source_class,
                        "error": repr(exc),
                    }
                )
                continue

            metadata_rows.append(
                {
                    "source_file": parquet_path.name,
                    "source_row": int(row_index),
                    "image_id": image_id,
                    "source_class": source_class,
                    "binary_label": label,
                    **stats,
                }
            )
            label_rows.append(
                {
                    "image_path": str(Path("artifacts") / "task01" / "cleaned_train" / "images" / cleaned_name),
                    "source_class": source_class,
                    "binary_label": label,
                    "original_width": stats["width"],
                    "original_height": stats["height"],
                    "cleaned_width": CLEANED_SIZE,
                    "cleaned_height": CLEANED_SIZE,
                }
            )

        log(f"  processed rows so far: {total_images}, valid: {len(metadata_rows)}, skipped: {len(skipped)}")

    metadata = pd.DataFrame(metadata_rows)
    labels = pd.DataFrame(label_rows)
    return metadata, labels, skipped, total_images


def main() -> None:
    args = parse_args()
    start = time.monotonic()
    deadline = start + max(args.timeout_seconds, 1) * 0.95

    data_dir = resolve_data_dir(args.data_dir)
    artifacts_dir = resolve_artifacts_dir(args.artifacts_dir)
    task_dir = ensure_dir(artifacts_dir / "task01")
    plots_dir = ensure_dir(task_dir / "plots")
    ensure_dir(task_dir / "cleaned_train")
    clean_previous_task01_outputs(task_dir)

    log(f"Using train data from: {data_dir / 'train'}")
    log(f"Writing Task 1.1 outputs to: {task_dir}")

    metadata, labels, skipped, total_images = process_train_split(data_dir / "train", task_dir, deadline)
    if metadata.empty:
        raise RuntimeError("No valid images were decoded from data/train.")

    metadata_path = task_dir / "train_metadata.csv"
    labels_path = task_dir / "cleaned_train" / "labels.csv"
    summary_path = task_dir / "summary.json"
    notes_path = task_dir / "cleaning_notes.txt"
    report_path = task_dir / "task01_report.md"

    log("Writing metadata and cleaned labels")
    metadata.to_csv(metadata_path, index=False)
    labels.to_csv(labels_path, index=False)

    log("Computing summary statistics")
    summary = summarize(metadata, total_images, skipped)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    log("Generating plots")
    create_plots(metadata, plots_dir)

    log("Writing cleaning notes")
    write_cleaning_notes(notes_path, summary)
    write_task01_report(report_path, summary)

    elapsed = time.monotonic() - start
    log("Task 1.1 complete")
    log(f"Total images: {total_images}")
    log(f"Valid images: {len(metadata)}")
    log(f"Corrupted/skipped images: {len(skipped)}")
    log(f"Elapsed time: {elapsed:.1f}s")
    log(f"Metadata CSV: {metadata_path}")
    log(f"Summary JSON: {summary_path}")
    log(f"Plots directory: {plots_dir}")
    log(f"Cleaned labels CSV: {labels_path}")
    log(f"Report markdown: {report_path}")


if __name__ == "__main__":
    main()
