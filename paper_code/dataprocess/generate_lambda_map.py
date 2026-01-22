"""Generate per-camera soft Gaussian change maps.

For each camera folder, we load all images and compute a per-pixel, per-channel
change statistic across the stack. The statistic follows the description:

    mu = sum_i y_i / N
    s  = sqrt( sum_i (y_i - mu) ** 2 )
    sigma = sqrt( mean_i (y_i - mu) ** 2 )
    score = exp( -s / (2 * t ** 2) )

Where `t` controls the steepness of the Gaussian response. The resulting score
map encodes how little a pixel/color channel changes across the stack: values
close to 1 indicate stable colors, while values near 0 mark strong variations.

The script writes, for each camera folder, a grayscale lambda map image (single
channel) as well as an `.npz` file containing the raw per-channel statistics
(`mu`, `s`, `sigma`) for further analysis.

Debug usage: pass `--debug-display` to preview the first computed lambda map in
matplotlib; the script exits once the window is closed.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


DEFAULT_DATASET_ROOT = Path("/media/barry/56EA40DEEA40BBCD/DATA/studio_test2/")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".exr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate soft Gaussian lambda maps per camera folder."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing per-camera folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where results are written. "
            "Defaults to <dataset-root>/lambda_maps."
        ),
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="cam",
        help="Substring that camera folders must contain.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default="05",
        help="Substring that camera folders must NOT contain.",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=0.02,
        help="Gaussian steepness parameter t. Smaller values create sharper masks.",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=2,
        help="Skip folders with fewer than this many images.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cameras whose outputs already exist.",
    )
    parser.add_argument(
        "--debug-display",
        action="store_true",
        help=(
            "Show the first generated lambda map using matplotlib and exit after "
            "the window is closed."
        ),
    )
    return parser.parse_args()


def list_camera_folders(
    root: Path, include: str, exclude: str
) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    folders = [
        entry
        for entry in sorted(root.iterdir())
        if entry.is_dir()
        and include in entry.name
        and (exclude not in entry.name if exclude else True)
    ]
    return folders


def list_images(folder: Path) -> List[Path]:
    return [
        entry
        for entry in sorted(folder.iterdir())
        if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
    ]


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")

    # Convert grayscale images to RGB by replicating the channel.
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    raw_dtype = image.dtype
    if raw_dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif raw_dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    elif raw_dtype in (np.float32, np.float64):
        image = image.astype(np.float32)
    else:
        raise ValueError(f"Unsupported image dtype {raw_dtype} for {path}")

    # cv2 loads images as BGR; convert to RGB for consistency.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def stack_images(paths: Sequence[Path]) -> np.ndarray:
    images = [read_image(path) for path in paths]
    shapes = {img.shape for img in images}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent image shapes in stack: {shapes}")
    return np.stack(images, axis=0)


def compute_statistics(
    stack: np.ndarray, t: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if stack.ndim != 4:
        raise ValueError(f"Expected stack of shape (N, H, W, C), got {stack.shape}")
    if t <= 0 or not math.isfinite(t):
        raise ValueError(f"Parameter t must be positive and finite, got {t}")

    mu = np.mean(stack, axis=0)
    diffs = stack - mu
    sum_sq = np.sum(np.square(diffs), axis=0)
    s = np.sqrt(sum_sq)
    sigma = np.sqrt(np.mean(np.square(diffs), axis=0))
    score = np.exp(-s / (2.0 * (t ** 2)))
    grayscale_score = np.mean(score, axis=2)
    return grayscale_score, score, mu, s, sigma


def save_outputs(
    output_root: Path,
    camera_name: str,
    grayscale_score: np.ndarray,
    per_channel_score: np.ndarray,
    mu: np.ndarray,
    s: np.ndarray,
    sigma: np.ndarray,
) -> None:
    camera_output = output_root / camera_name
    camera_output.mkdir(parents=True, exist_ok=True)

    # Clamp score to [0, 1] and convert to 8-bit grayscale image.
    score_img = np.clip(grayscale_score, 0.0, 1.0)
    score_img = (score_img * 255.0).round().astype(np.uint8)
    image_path = camera_output / "lambda_map.png"
    cv2.imwrite(str(image_path), score_img)

    stats_path = camera_output / "lambda_stats.npz"
    np.savez_compressed(
        stats_path,
        mu=mu,
        s=s,
        sigma=sigma,
        per_channel_score=per_channel_score,
        grayscale_score=grayscale_score,
    )


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root.resolve()
    output_root: Path = (
        args.output_dir.resolve()
        if args.output_dir
        else (dataset_root / "lambda_maps").resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)

    camera_folders = list_camera_folders(dataset_root, args.pattern, args.exclude)
    if not camera_folders:
        raise RuntimeError(f"No camera folders found under {dataset_root}")

    for index, folder in enumerate(camera_folders):
        image_paths = list_images(folder)
        if len(image_paths) < args.min_images:
            print(
                f"[skip] {folder.name}: only {len(image_paths)} eligible image(s); "
                f"need at least {args.min_images}."
            )
            continue

        output_exists = (
            output_root / folder.name / "lambda_map.png"
        ).exists()
        if args.skip_existing and output_exists:
            print(f"[skip] {folder.name}: outputs already exist.")
            continue

        print(f"[process] {folder.name}: {len(image_paths)} images.")
        stack = stack_images(image_paths)
        grayscale_score, per_channel_score, mu, s, sigma = compute_statistics(
            stack, t=args.t
        )
        save_outputs(
            output_root,
            folder.name,
            grayscale_score,
            per_channel_score,
            mu,
            s,
            sigma,
        )

        if args.debug_display:
            try:
                import matplotlib.pyplot as plt
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Debug display requested but matplotlib is not installed."
                ) from exc

            plt.figure(figsize=(8, 6))
            plt.imshow(grayscale_score, cmap="gray", vmin=0.0, vmax=1.0)
            plt.title(f"Lambda Map - {folder.name}")
            plt.colorbar(label="Soft Gaussian Score")
            plt.tight_layout()
            print(f"[debug] Displaying lambda map for {folder.name}. Close the window to exit.")
            plt.show()
            print("[debug] Exiting after debug display.")
            return

    print(f"[done] Lambda maps saved under {output_root}")


if __name__ == "__main__":
    main()
