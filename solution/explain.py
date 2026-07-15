#!/usr/bin/env python3
"""
Use Grad-CAM to explain our final augmented AI-image detector.
This runs entirely on CPU, reads both validation parquet splits without modifying them,
and saves all generated visualizations and reports under ``artifacts/task04``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable

# Keep matplotlib caching self-contained. Writing to the default home directory 
# can cause permission issues in restricted or Docker-like environments.
_PLOT_CACHE = Path(__file__).resolve().parent / "artifacts" / "task04" / ".plot_cache"
_PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_PLOT_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_PLOT_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Subset

from train import ParquetCalibDataset
from train import ai_probability
from train_augmented import StrongerCNNDetector, resolve_data_dir as pipeline_resolve_data_dir


IMAGE_SIZE = 128
DISPLAY_SIZE = 224
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
SPLIT_NAMES = ("validation", "validation_augmented")
RESULT_COLUMNS = [
    "split",
    "sample_index",
    "source_class",
    "true_label",
    "predicted_label",
    "ai_score",
    "classification_threshold",
    "correct",
    "error_category",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout_seconds", type=int, default=600)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Deterministic maximum per split (mainly useful for smoke tests).",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=11,
        help="Maximum number of representative examples to visualize.",
    )
    parser.add_argument("--attention_samples_per_class", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save_true_class_errors",
        action="store_true",
        help="Also save true-class Grad-CAM overlays for misclassified examples.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def clean_previous_outputs(task_dir: Path, figures_dir: Path) -> None:
    """Clears out files from earlier Task 1.4 runs so we don't mix up old and new results."""
    generated_names = [
        "explanation_results.csv",
        "metrics_summary.json",
        "selected_examples.csv",
        "perturbation_results.csv",
        "attention_statistics.csv",
        "report_summary.md",
    ]
    for name in generated_names:
        path = task_dir / name
        if path.is_file():
            path.unlink()
    for path in figures_dir.glob("*.png"):
        path.unlink()


def set_deterministic(seed: int) -> None:
    """it forces CPU-only execution and locks down all random seeds for full reproducibility."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def evaluation_transform() -> transforms.Compose:
    """it aplies the exact same evaluation transform we defined in Task 1.3."""
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def resolve_split_dir(name: str, script_dir: Path) -> Path:
    """Finds the dataset split folder, prioritizing project-local paths first."""
    pipeline_path = Path(pipeline_resolve_data_dir(name))
    candidates = [
        script_dir / "data" / name,
        script_dir.parent / "data" / name,
        script_dir.parent / "AML DATA" / name,
        pipeline_path,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.parquet")):
            return candidate.resolve()
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find parquet data for {name}. Tried: {tried}")


def load_final_model(checkpoint: Path) -> StrongerCNNDetector:
   """it loads our final Task 1.3 model architecture and maps the saved weights to CPU."""
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Task 1.3 checkpoint is missing: {checkpoint}. Run train_augmented.py first."
        )
    model = StrongerCNNDetector(channels=32, dropout=0.3).cpu()
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older project PyTorch versions.
        state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def load_saved_threshold(path: Path) -> float:
    """Loads our calibrated decision threshold from Task 1.3.

    We reuse this threshold directly; this script does not re-calibrate it.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Task 1.3 calibrated threshold is missing: {path}. "
            "No replacement threshold is introduced by explain.py."
        )
    threshold = float(path.read_text(encoding="utf-8").strip())
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid calibrated threshold {threshold!r} in {path}")
    return threshold


def deterministic_indices(length: int, maximum: int | None) -> list[int]:
    if maximum is None or maximum >= length:
        return list(range(length))
    if maximum <= 0:
        raise ValueError("--max_samples must be positive")
    return np.linspace(0, length - 1, maximum, dtype=int).tolist()


def category_name(true_label: int, prediction: int) -> str:
    return {
        (1, 1): "true positive",
        (0, 0): "true negative",
        (0, 1): "false positive",
        (1, 0): "false negative",
    }[(true_label, prediction)]


def save_results(rows: list[dict], output_path: Path) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    result.to_csv(output_path, index=False)
    return result


def evaluate_split(
    model: nn.Module,
    split_name: str,
    dataset: ParquetCalibDataset,
    threshold: float,
    batch_size: int,
    maximum: int | None,
    deadline: float,
    rows: list[dict],
    partial_csv: Path,
) -> bool:
    
    indices = deterministic_indices(len(dataset), maximum)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    offset = 0
    for batch_number, (images, labels) in enumerate(loader, start=1):
        if time.monotonic() >= deadline:
            log(f"[TIMEOUT] Stopping {split_name} evaluation at a batch boundary.")
            save_results(rows, partial_csv)
            return False
        with torch.no_grad():
            scores = ai_probability(model(images.cpu())).cpu().numpy()
        label_values = labels.cpu().numpy().astype(int)
        batch_indices = indices[offset : offset + len(label_values)]
        for local_index, true_label, score in zip(batch_indices, label_values, scores):
            prediction = int(float(score) >= threshold)
            source_class = int(dataset.main_df.iloc[local_index]["source_class"])
            rows.append(
                {
                    "split": split_name,
                    "sample_index": int(local_index),
                    "source_class": source_class,
                    "true_label": int(true_label),
                    "predicted_label": prediction,
                    "ai_score": float(score),
                    "classification_threshold": threshold,
                    "correct": bool(prediction == true_label),
                    "error_category": category_name(int(true_label), prediction),
                }
            )
        offset += len(label_values)
        if batch_number % 10 == 0:
            save_results(rows, partial_csv)
            log(f"  {split_name}: evaluated {offset}/{len(indices)} samples")
    save_results(rows, partial_csv)
    log(f"  {split_name}: evaluated {offset}/{len(indices)} samples")
    return True


def safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def metrics_for(group: pd.DataFrame, expected: int) -> dict:
    labels = group["true_label"].to_numpy(dtype=int)
    predictions = group["predicted_label"].to_numpy(dtype=int)
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return {
        "number_of_samples": int(len(group)),
        "expected_samples": int(expected),
        "partial": bool(len(group) < expected),
        "accuracy": safe_ratio(tp + tn, len(group)),
        "ai_recall": safe_ratio(tp, tp + fn),
        "false_positive_rate_on_real": safe_ratio(fp, fp + tn),
        "precision": safe_ratio(tp, tp + fp),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
    }


def save_metrics(
    results: pd.DataFrame,
    datasets: dict[str, ParquetCalibDataset],
    maximum: int | None,
    output_path: Path,
) -> dict:
    summary = {}
    for split in SPLIT_NAMES:
        expected = len(datasets[split])
        summary[split] = metrics_for(results.loc[results["split"] == split], expected)
        summary[split]["requested_sample_cap"] = maximum
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def choose_balanced(candidates: pd.DataFrame, count: int) -> list[int]:
    """Round-robin already-ranked candidates across the two validation splits."""
    if count <= 0 or candidates.empty:
        return []
    by_split = {
        split: candidates.loc[candidates["split"] == split].index.tolist() for split in SPLIT_NAMES
    }
    chosen: list[int] = []
    while len(chosen) < count:
        changed = False
        for split in SPLIT_NAMES:
            if by_split[split] and len(chosen) < count:
                chosen.append(by_split[split].pop(0))
                changed = True
        if not changed:
            break
    return chosen


def select_examples(results: pd.DataFrame, maximum: int) -> pd.DataFrame:
   """Pick a balanced, representative mix of high-confidence predictions, errors, and borderline cases."""
    if results.empty or maximum <= 0:
        return results.iloc[0:0].copy()
    work = results.copy()
    work["distance_to_threshold"] = (
        work["ai_score"] - work["classification_threshold"]
    ).abs()
    specifications = [
        ("true positive", 2, "ai_score", False),
        ("true negative", 2, "ai_score", True),
        ("false positive", 3, "ai_score", False),
        ("false negative", 3, "ai_score", True),
    ]
    selected: list[int] = []
    reasons: dict[int, str] = {}
    for category, quota, column, ascending in specifications:
        candidates = work.loc[
            (work["error_category"] == category) & (~work.index.isin(selected))
        ].sort_values([column, "split", "sample_index"], ascending=[ascending, True, True])
        for index in choose_balanced(candidates, min(quota, maximum - len(selected))):
            selected.append(index)
            reasons[index] = f"confident {category}"
        if len(selected) >= maximum:
            break

    if len(selected) < maximum:
        selected_categories = work.loc[selected, "error_category"].value_counts()
        error_limits = {"false positive": 3, "false negative": 3}
        eligible = ~work.index.isin(selected)
        for category, limit in error_limits.items():
            if int(selected_categories.get(category, 0)) >= limit:
                eligible &= work["error_category"] != category
        uncertain = work.loc[eligible].sort_values(
            ["distance_to_threshold", "split", "sample_index"]
        )
        uncertain_quota = min(maximum - len(selected), max(1, min(2, len(uncertain))))
        for index in choose_balanced(uncertain, uncertain_quota):
            selected.append(index)
            reasons[index] = "uncertain near threshold"

    if len(selected) < maximum:
        selected_categories = work.loc[selected, "error_category"].value_counts()
        eligible = ~work.index.isin(selected)
        for category, limit in {"false positive": 3, "false negative": 3}.items():
            if int(selected_categories.get(category, 0)) >= limit:
                eligible &= work["error_category"] != category
        remaining = work.loc[eligible].sort_values(
            ["distance_to_threshold", "split", "sample_index"]
        )
        for index in remaining.index[: maximum - len(selected)]:
            selected.append(index)
            reasons[index] = "additional informative example"

    output = work.loc[selected].copy()
    output["selection_reason"] = [reasons[index] for index in selected]
    return output


def find_last_convolution(model: nn.Module) -> tuple[str, nn.Conv2d]:
    convolutions = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]
    if not convolutions:
        raise RuntimeError("The Task 1.3 model has no convolutional feature map for Grad-CAM.")
    return convolutions[-1]


class GradCAM:
    """Grad-CAM implementation using standard forward and backward hooks on the target conv layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module, threshold: float):
        self.model = model
        self.threshold = threshold
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def explain(
        self, image_tensor: torch.Tensor, target_class: int | None = None, output_size=(DISPLAY_SIZE, DISPLAY_SIZE)
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Return normalized CAM, logits, and explained class (predicted by default)."""
        self.activations = None
        self.gradients = None
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor.unsqueeze(0).cpu())
        binary_score = ai_probability(logits)[0]
        explained_class = (
            int(float(binary_score.detach()) >= self.threshold)
            if target_class is None
            else int(target_class)
        )
        
        ai_logit = torch.logsumexp(logits[0, 1:], dim=0)
        binary_logit = ai_logit - logits[0, 0]
        (binary_logit if explained_class == 1 else -binary_logit).backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)[0, 0]
        minimum, maximum = cam.min(), cam.max()
        if float(maximum - minimum) > 1e-12:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = torch.zeros_like(cam)
        return cam.cpu().numpy(), logits.detach().cpu().numpy()[0], explained_class

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def original_image(dataset: ParquetCalibDataset, index: int) -> Image.Image:
    raw_bytes = dataset.main_df.iloc[index]["image"]
    with Image.open(BytesIO(raw_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.BICUBIC)


def preprocess_display_image(image: Image.Image) -> torch.Tensor:
    return evaluation_transform()(image)


def window_scores(heatmap: np.ndarray, height: int, width: int) -> np.ndarray:
    integral = np.pad(heatmap, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    return (
        integral[height:, width:]
        - integral[:-height, width:]
        - integral[height:, :-width]
        + integral[:-height, :-width]
    )


def salient_and_control_boxes(heatmap: np.ndarray, fraction: float = 0.25) -> tuple[tuple, tuple]:
    height, width = heatmap.shape
    box_h, box_w = max(1, round(height * fraction)), max(1, round(width * fraction))
    scores = window_scores(heatmap, box_h, box_w)
    salient_y, salient_x = np.unravel_index(int(np.argmax(scores)), scores.shape)
    control_y, control_x = np.unravel_index(int(np.argmin(scores)), scores.shape)
    salient = (int(salient_x), int(salient_y), int(salient_x + box_w), int(salient_y + box_h))
    control = (int(control_x), int(control_y), int(control_x + box_w), int(control_y + box_h))
    return salient, control


def occlude(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Replace a region with the ImageNet mean, which maps approximately to normalized zero."""
    output = image.copy()
    mean_rgb = tuple(int(round(value * 255)) for value in NORMALIZE_MEAN)
    output.paste(Image.new("RGB", (box[2] - box[0], box[3] - box[1]), mean_rgb), box[:2])
    return output


def ai_score(model: nn.Module, image: Image.Image) -> float:
    with torch.no_grad():
        logits = model(preprocess_display_image(image).unsqueeze(0).cpu())
        return float(ai_probability(logits)[0].item())


def class_label(value: int) -> str:
    return "AI-generated" if int(value) == 1 else "Real"


def heatmap_rgb(heatmap: np.ndarray) -> np.ndarray:
    return plt.get_cmap("magma")(heatmap)[..., :3]


def overlay_rgb(image: Image.Image, heatmap: np.ndarray) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32) / 255.0
    return np.clip(0.58 * base + 0.42 * heatmap_rgb(heatmap), 0.0, 1.0)


def save_example_figure(row: pd.Series, image: Image.Image, heatmap: np.ndarray, perturbed: Image.Image, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(10.8, 3.1))
    panels = [
        (image, "Original"),
        (heatmap, "Grad-CAM"),
        (overlay_rgb(image, heatmap), "Overlay"),
        (perturbed, "Salient region occluded"),
    ]
    for axis, (content, title) in zip(axes, panels):
        axis.imshow(content, cmap="magma" if title == "Grad-CAM" else None, vmin=0, vmax=1)
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    fig.suptitle(
        f"{row['split']} #{int(row['sample_index'])} | true: {class_label(row['true_label'])} | "
        f"predicted: {class_label(row['predicted_label'])} | AI score: {row['ai_score']:.3f} | "
        f"threshold: {row['classification_threshold']:.3f} | {row['error_category']}",
        fontsize=9,
    )
    fig.text(
        0.5,
        0.015,
        "Predicted-class logit; Grad-CAM normalized per image to [0,1]; "
        "gray patch = ImageNet-mean occlusion.",
        ha="center",
        fontsize=7.5,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_true_class_figure(row: pd.Series, image: Image.Image, heatmap: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.6))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(overlay_rgb(image, heatmap))
    axes[1].set_title(f"True-class CAM: {class_label(row['true_label'])}")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_panel(records: list[dict], path: Path, title: str, categories: Iterable[str]) -> None:
    category_list = list(categories)
    groups: list[tuple[str, list[dict]]] = []
    for category in category_list:
        candidates = [record for record in records if record["row"]["error_category"] == category]
        chosen: list[dict] = []
        # Prefer one example from each split, then fill any remaining slot.
        for split in SPLIT_NAMES:
            match = next((record for record in candidates if record["row"]["split"] == split), None)
            if match is not None and all(match is not existing for existing in chosen):
                chosen.append(match)
        for record in candidates:
            if len(chosen) >= 2:
                break
            if all(record is not existing for existing in chosen):
                chosen.append(record)
        if chosen:
            groups.append((category, chosen[:2]))

    if not groups:
        fig, axis = plt.subplots(figsize=(6, 2))
        axis.text(0.5, 0.5, "No matching examples were available.", ha="center", va="center")
        axis.axis("off")
    else:
        fig, axes = plt.subplots(len(groups), 4, figsize=(10.2, 3.8 * len(groups)), squeeze=False)
        for row_axes, (_category, chosen) in zip(axes, groups):
            for example_number, record in enumerate(chosen):
                original_axis = row_axes[2 * example_number]
                overlay_axis = row_axes[2 * example_number + 1]
                original_axis.imshow(record["image"])
                original_axis.set_title(
                    f"Original: {record['row']['split']}\n"
                    f"#{int(record['row']['sample_index'])} | true {class_label(record['row']['true_label'])}\n"
                    f"pred {class_label(record['row']['predicted_label'])}",
                    fontsize=7.5,
                )
                overlay_axis.imshow(overlay_rgb(record["image"], record["heatmap"]))
                overlay_axis.set_title(
                    f"{record['row']['error_category']}\n"
                    f"AI score={record['row']['ai_score']:.3f}\n"
                    f"threshold={record['row']['classification_threshold']:.3f}",
                    fontsize=7.5,
                )
            for axis in row_axes:
                axis.axis("off")
    fig.suptitle(
        f"{title}\nPredicted-class Grad-CAM; each heatmap normalized independently to [0,1]",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=4.5, w_pad=1.2)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def attention_measurements(heatmap: np.ndarray) -> dict[str, float]:
    """Compute descriptive concentration, entropy, and center-mass measures."""
    values = np.asarray(heatmap, dtype=np.float64).clip(min=0)
    total = float(values.sum())
    if total <= 1e-12:
        return {"top_20_percent_mass": 0.0, "normalized_entropy": 0.0, "center_mass": 0.0}
    probabilities = values.ravel() / total
    top_count = max(1, int(math.ceil(probabilities.size * 0.20)))
    top_mass = float(np.partition(probabilities, -top_count)[-top_count:].sum())
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum() / math.log(probabilities.size))
    height, width = values.shape
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    center_mass = float(values[y0:y1, x0:x1].sum() / total)
    return {
        "top_20_percent_mass": top_mass,
        "normalized_entropy": entropy,
        "center_mass": center_mass,
    }


def plot_perturbations(perturbations: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.4, 3.6))
    if perturbations.empty:
        axis.text(0.5, 0.5, "No perturbation results available.", ha="center", va="center")
        axis.axis("off")
    else:
        positions = np.arange(len(perturbations))
        width = 0.38
        axis.bar(
            positions - width / 2,
            perturbations["absolute_score_change_salient"],
            width,
            label="salient region",
            color="#e15759",
        )
        axis.bar(
            positions + width / 2,
            perturbations["absolute_score_change_control"],
            width,
            label="low-saliency control",
            color="#4e79a7",
        )
        axis.set_xticks(positions)
        axis.set_xticklabels(
            [f"{split.replace('validation', 'val')}\n#{index}" for split, index in zip(perturbations["split"], perturbations["sample_index"])],
            rotation=35,
            ha="right",
            fontsize=7,
        )
        axis.set_ylabel("Absolute change in AI probability")
        axis.legend(fontsize=8)
    axis.set_title(f"Grad-CAM perturbation sanity check (n={len(perturbations)})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_attention_statistics(statistics: pd.DataFrame, output_path: Path) -> None:
    measures = ["top_20_percent_mass", "normalized_entropy", "center_mass"]
    labels = ["Top-20% mass", "Normalized entropy", "Center mass"]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0))
    for axis, measure, label in zip(axes, measures, labels):
        groups = [
            statistics.loc[statistics["true_label"] == class_value, measure].dropna().to_numpy()
            for class_value in (0, 1)
        ]
        if all(len(group) for group in groups):
            axis.boxplot(groups, tick_labels=["Real", "AI-generated"], showfliers=False)
        else:
            axis.text(0.5, 0.5, "insufficient data", ha="center", va="center")
            axis.set_xticks([])
        axis.set_title(label, fontsize=9)
        axis.tick_params(axis="x", labelrotation=20, labelsize=8)
    real_n = int((statistics["true_label"] == 0).sum()) if "true_label" in statistics else 0
    ai_n = int((statistics["true_label"] == 1).sum()) if "true_label" in statistics else 0
    fig.suptitle(
        f"Descriptive Grad-CAM statistics for correct predictions (real n={real_n}, AI n={ai_n})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def deterministic_attention_subset(results: pd.DataFrame, per_class: int) -> pd.DataFrame:
    correct = results.loc[results["correct"]].copy()
    chosen: list[int] = []
    for label in (0, 1):
        candidates = correct.loc[correct["true_label"] == label].sort_values(["split", "sample_index"])
        if len(candidates) > per_class:
            positions = np.linspace(0, len(candidates) - 1, per_class, dtype=int)
            candidates = candidates.iloc[positions]
        chosen.extend(candidates.index.tolist())
    return correct.loc[chosen]


def analyze_selected(
    model: nn.Module,
    cam: GradCAM,
    selected: pd.DataFrame,
    datasets: dict[str, ParquetCalibDataset],
    figures_dir: Path,
    save_true_errors: bool,
    deadline: float,
) -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    visual_records: list[dict] = []
    for _, row in selected.iterrows():
        if time.monotonic() >= deadline:
            log("[TIMEOUT] Stopping representative-example analysis.")
            break
        split, index = str(row["split"]), int(row["sample_index"])
        dataset = datasets[split]
        image = original_image(dataset, index)
        tensor, _ = dataset[index]
        heatmap, _, _ = cam.explain(tensor, target_class=None)
        salient_box, control_box = salient_and_control_boxes(heatmap)
        salient_image = occlude(image, salient_box)
        control_image = occlude(image, control_box)
        salient_score = ai_score(model, salient_image)
        control_score = ai_score(model, control_image)
        original_score = float(row["ai_score"])
        rows.append(
            {
                "split": split,
                "sample_index": index,
                "true_label": int(row["true_label"]),
                "original_prediction": int(row["predicted_label"]),
                "original_ai_score": original_score,
                "score_after_salient_region_perturbation": salient_score,
                "score_change_after_salient_region_perturbation": original_score - salient_score,
                "score_after_control_region_perturbation": control_score,
                "score_change_after_control_region_perturbation": original_score - control_score,
                "absolute_score_change_salient": abs(original_score - salient_score),
                "absolute_score_change_control": abs(original_score - control_score),
            }
        )
        safe_category = str(row["error_category"]).replace(" ", "_")
        stem = f"{split}_{index:04d}_{safe_category}"
        save_example_figure(row, image, heatmap, salient_image, figures_dir / f"{stem}.png")
        if save_true_errors and not bool(row["correct"]):
            true_heatmap, _, _ = cam.explain(tensor, target_class=int(row["true_label"]))
            save_true_class_figure(row, image, true_heatmap, figures_dir / f"{stem}_true_class.png")
        visual_records.append({"row": row, "image": image, "heatmap": heatmap})
        log(f"  explained {split} sample {index} ({row['error_category']})")
    return pd.DataFrame(
        rows,
        columns=[
            "split",
            "sample_index",
            "true_label",
            "original_prediction",
            "original_ai_score",
            "score_after_salient_region_perturbation",
            "score_change_after_salient_region_perturbation",
            "score_after_control_region_perturbation",
            "score_change_after_control_region_perturbation",
            "absolute_score_change_salient",
            "absolute_score_change_control",
        ],
    ), visual_records


def compute_attention_statistics(
    cam: GradCAM,
    results: pd.DataFrame,
    datasets: dict[str, ParquetCalibDataset],
    per_class: int,
    deadline: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    subset = deterministic_attention_subset(results, max(1, per_class))
    for _, row in subset.iterrows():
        if time.monotonic() >= deadline:
            log("[TIMEOUT] Saving partial attention statistics.")
            break
        split, index = str(row["split"]), int(row["sample_index"])
        tensor, _ = datasets[split][index]
        heatmap, _, _ = cam.explain(tensor, target_class=None)
        rows.append(
            {
                "split": split,
                "sample_index": index,
                "true_label": int(row["true_label"]),
                "class_name": class_label(int(row["true_label"])),
                **attention_measurements(heatmap),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "split",
            "sample_index",
            "true_label",
            "class_name",
            "top_20_percent_mass",
            "normalized_entropy",
            "center_mass",
        ],
    )


def median_or_none(frame: pd.DataFrame, column: str) -> float | None:
    return float(frame[column].median()) if not frame.empty else None


def fmt(value: float | None) -> str:
    return "not available" if value is None or not math.isfinite(value) else f"{value:.3f}"


def fmt_percent(value: float | None) -> str:
    return "not available" if value is None or not math.isfinite(value) else f"{value:.1%}"


def write_report(
    path: Path,
    results: pd.DataFrame,
    metrics: dict,
    selected: pd.DataFrame,
    perturbations: pd.DataFrame,
    attention: pd.DataFrame,
    layer_name: str,
    threshold: float,
) -> None:
    real_attention = attention.loc[attention["true_label"] == 0]
    ai_attention = attention.loc[attention["true_label"] == 1]
    fp = results.loc[results["error_category"] == "false positive"]
    fn = results.loc[results["error_category"] == "false negative"]
    if perturbations.empty:
        faithfulness = "No perturbation comparisons were completed before the runtime limit."
    else:
        salient_abs = float(perturbations["absolute_score_change_salient"].mean())
        control_abs = float(perturbations["absolute_score_change_control"].mean())
        share = float(
            (perturbations["absolute_score_change_salient"] > perturbations["absolute_score_change_control"]).mean()
        )
        relation = "larger" if salient_abs > control_abs else "not larger"
        faithfulness = (
            f"Mean absolute AI-score change was {salient_abs:.3f} for salient occlusion and "
            f"{control_abs:.3f} for the low-saliency control. The salient effect was {relation} on average "
            f"and exceeded the control in {share:.1%} of {len(perturbations)} comparisons. "
            "This provides partial faithfulness evidence, but the small changes and inconsistent per-image "
            "advantage do not validate every highlighted region."
        )

    attention_text = (
        f"For {len(real_attention)} correctly classified real and {len(ai_attention)} correctly classified "
        f"AI-generated examples, the median fraction of saliency mass in the most salient 20% "
        f"of pixels was {fmt(median_or_none(real_attention, 'top_20_percent_mass'))} for real images and "
        f"{fmt(median_or_none(ai_attention, 'top_20_percent_mass'))} for AI-generated images. Median normalized "
        f"entropy was {fmt(median_or_none(real_attention, 'normalized_entropy'))} versus "
        f"{fmt(median_or_none(ai_attention, 'normalized_entropy'))}, and median center mass was "
        f"{fmt(median_or_none(real_attention, 'center_mass'))} versus "
        f"{fmt(median_or_none(ai_attention, 'center_mass'))}. These are descriptive differences, not causal proof."
    )
    fp_aug = safe_ratio(int((fp["split"] == "validation_augmented").sum()), len(fp))
    fn_aug = safe_ratio(int((fn["split"] == "validation_augmented").sum()), len(fn))
    full_run = all(not bool(metrics[split]["partial"]) for split in SPLIT_NAMES)
    if full_run:
        visual_error_observation = (
            "In the deterministic full-run panels, confident false positives include visually unusual or "
            "staged real scenes with strong saturation, blur, or striking composition. Selected false negatives "
            "include photorealistic synthetic architecture, interiors, and dense city scenes. Their maps often "
            "emphasize boundaries or isolated structures rather than an obvious synthetic defect."
        )
    else:
        visual_error_observation = (
            "The error panels permit qualitative inspection, but a capped run should not be used to infer "
            "full-split visual patterns."
        )
    error_text = (
        f"There were {len(fp)} false positives and {len(fn)} false negatives in the evaluated rows. "
        f"False positives had median AI score {fmt(median_or_none(fp, 'ai_score'))}; "
        f"{fmt_percent(fp_aug)} of them came from validation_augmented. False negatives had median AI score "
        f"{fmt(median_or_none(fn, 'ai_score'))}; {fmt_percent(fn_aug)} came from validation_augmented. "
        f"{visual_error_observation} These observations are plausible shortcut hypotheses, not automatically "
        "verified semantic explanations."
    )
    metric_rows = []
    for split in SPLIT_NAMES:
        item = metrics[split]
        metric_rows.append(
            f"| {split} | {item['number_of_samples']} | {fmt(item['accuracy'])} | "
            f"{fmt(item['ai_recall'])} | {fmt(item['false_positive_rate_on_real'])} | "
            f"{item['false_positives']} | {item['false_negatives']} |"
        )
    validation_metrics = metrics["validation"]
    augmented_metrics = metrics["validation_augmented"]
    operating_point_text = (
        f"The relatively conservative threshold protects real images: FPR was "
        f"{fmt_percent(validation_metrics['false_positive_rate_on_real'])} on validation and "
        f"{fmt_percent(augmented_metrics['false_positive_rate_on_real'])} on validation_augmented, while "
        f"precision remained {fmt_percent(validation_metrics['precision'])} and "
        f"{fmt_percent(augmented_metrics['precision'])}. The cost is a false-negative-heavy error profile. "
        f"AI recall fell from {fmt_percent(validation_metrics['ai_recall'])} to "
        f"{fmt_percent(augmented_metrics['ai_recall'])}, with false negatives increasing from "
        f"{validation_metrics['false_negatives']} to {augmented_metrics['false_negatives']}."
    )
    report = f"""# Task 1.4 Explainability Summary

## Method and model

The final Task 1.3 `StrongerCNNDetector` checkpoint was explained with Grad-CAM at convolutional layer
`{layer_name}`. Grad-CAM is appropriate because this CNN retains spatial convolutional feature maps before
global average pooling. Each displayed map uses the aggregate binary AI-versus-real logit (combining source
classes 1--5), ReLU, per-image normalization, and bilinear resizing. The saved calibrated Task 1.3 threshold
was {threshold:.6f}.

## Evaluation

| split | n | accuracy | AI recall | real-image FPR | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

`source_class == 0` was treated as real; source classes 1--5 were merged as AI-generated. Results marked
partial in `metrics_summary.json` should not be treated as full-split estimates.

{operating_point_text}

## Representative examples

The deterministic selection requested confident true positives and true negatives, available false positives
and false negatives, then predictions closest to the calibrated threshold. It alternated between validation
splits when both supplied a category. {len(selected)} examples were selected and {len(perturbations)} received
the complete Grad-CAM and perturbation analysis.

## Real versus AI-generated attention

{attention_text}

The accompanying panels show comparatively diffuse predicted-class attention over correctly classified
AI-generated subjects/scenes, whereas the selected real-image maps are more localized and often include image
boundaries. This visual pattern agrees with the concentration and entropy summaries, but it does not show that
the highlighted content caused the prediction.

## False positives and false negatives

{error_text}

## Perturbation sanity check

{faithfulness}

Occlusion used an equally sized low-saliency region as a control. Changes in AI probability can have opposite
directions for real-class and AI-class explanations, so absolute changes are used for the aggregate comparison.

## Plausibility, shortcuts, and dataset bias

An explanation is more plausible when highlighted regions are visually relevant and salient-region occlusion
has a larger effect than the control. Even then, the result is only a basic faithfulness check. Center-heavy or
highly concentrated maps may indicate reliance on dominant subjects, while diffuse or border-heavy maps may
indicate texture, compression, padding, or background cues. The attention statistics and panels should be read
together; ambiguous differences remain ambiguous.

A validation-versus-augmented performance gap or concentration of errors in `validation_augmented` is
consistent with sensitivity to distortions, although it does not identify the cause. Source classes also come
from different generators, and real and synthetic images may differ in content, compression, resolution history,
or editing pipeline. Consequently, the classifier may learn generator/dataset signatures instead of a general
concept of image authenticity. Resizing reduces original-dimension shortcuts but cannot remove all acquisition
or encoding bias.

## Limitations

Grad-CAM is low resolution at the final convolutional layer and can omit evidence, merge distinct regions, or
look convincing when it is not faithful. Per-image min-max normalization hides absolute attribution strength.
Occlusion creates out-of-distribution content and measures association with a region rather than human-like or
causal reasoning. Visual explanations therefore are not definitive proof of why the model decided, and this
descriptive subset cannot establish general behavior for all images or unseen generators.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout_seconds must be positive")
    start = time.monotonic()
    # Reserve a small window for persisting partial tables and the report.
    deadline = start + max(1.0, args.timeout_seconds * 0.94)
    set_deterministic(args.seed)

    script_dir = Path(__file__).resolve().parent
    task_dir = script_dir / "artifacts" / "task04"
    figures_dir = task_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    clean_previous_outputs(task_dir, figures_dir)
    checkpoint = script_dir / "artifacts" / "task03" / "best_model_augmented.pt"
    threshold_path = script_dir / "artifacts" / "task03" / "calibrated_threshold.txt"

    log("Loading final Task 1.3 augmented model on CPU")
    model = load_final_model(checkpoint)
    threshold = load_saved_threshold(threshold_path)
    layer_name, target_layer = find_last_convolution(model)
    log(f"Using checkpoint: {checkpoint}")
    log(f"Using calibrated threshold: {threshold:.6f}")
    log(f"Using Grad-CAM layer: {layer_name}")

    datasets: dict[str, ParquetCalibDataset] = {}
    for split in SPLIT_NAMES:
        data_dir = resolve_split_dir(split, script_dir)
        log(f"Loading {split} from {data_dir}")
        datasets[split] = ParquetCalibDataset(str(data_dir), img_transformer=evaluation_transform())
        if not len(datasets[split]):
            raise RuntimeError(f"No samples were loaded for {split} from {data_dir}")

    results_path = task_dir / "explanation_results.csv"
    rows: list[dict] = []
    completed = True
    for split in SPLIT_NAMES:
        if time.monotonic() >= deadline:
            completed = False
            break
        completed = evaluate_split(
            model,
            split,
            datasets[split],
            threshold,
            max(1, args.batch_size),
            args.max_samples,
            deadline,
            rows,
            results_path,
        ) and completed
    results = save_results(rows, results_path)
    if results.empty:
        raise RuntimeError("No validation examples were evaluated before the timeout.")
    metrics = save_metrics(results, datasets, args.max_samples, task_dir / "metrics_summary.json")

    selected = select_examples(results, max(1, args.num_examples))
    selected.to_csv(task_dir / "selected_examples.csv", index=False)
    perturbations = pd.DataFrame()
    attention = pd.DataFrame()
    visual_records: list[dict] = []
    with GradCAM(model, target_layer, threshold) as cam:
        perturbations, visual_records = analyze_selected(
            model,
            cam,
            selected,
            datasets,
            figures_dir,
            args.save_true_class_errors,
            deadline,
        )
        perturbations.to_csv(task_dir / "perturbation_results.csv", index=False)
        if time.monotonic() < deadline:
            attention = compute_attention_statistics(
                cam,
                results,
                datasets,
                args.attention_samples_per_class,
                deadline,
            )
    if attention.empty:
        attention = pd.DataFrame(
            columns=[
                "split",
                "sample_index",
                "true_label",
                "class_name",
                "top_20_percent_mass",
                "normalized_entropy",
                "center_mass",
            ]
        )
    attention.to_csv(task_dir / "attention_statistics.csv", index=False)

    save_panel(
        visual_records,
        figures_dir / "error_examples.png",
        "Representative false positives and false negatives",
        ("false positive", "false negative"),
    )
    save_panel(
        visual_records,
        figures_dir / "real_vs_ai_attention.png",
        "Correctly classified real versus AI-generated attention",
        ("true negative", "true positive"),
    )
    if "absolute_score_change_salient" not in perturbations:
        perturbations = pd.DataFrame(
            columns=["split", "sample_index", "absolute_score_change_salient", "absolute_score_change_control"]
        )
    plot_perturbations(perturbations, figures_dir / "perturbation_summary.png")
    plot_attention_statistics(attention, figures_dir / "attention_statistics.png")
    write_report(
        task_dir / "report_summary.md",
        results,
        metrics,
        selected,
        perturbations,
        attention,
        layer_name,
        threshold,
    )

    elapsed = time.monotonic() - start
    log(f"Task 1.4 {'complete' if completed else 'saved with partial evaluation'} in {elapsed:.1f}s")
    log(f"Evaluated rows: {len(results)}")
    log(f"Outputs: {task_dir}")


if __name__ == "__main__":
    main()
