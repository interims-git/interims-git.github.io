import argparse
import os
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def list_cameras(root: Path, start: int = 0, end: int = 15):
    """Yield camera IDs like cam00..cam15 that exist under images/."""
    for idx in range(start, end + 1):
        cam = f"cam{idx:02d}"
        cam_dir = root / "images" / cam
        if cam_dir.is_dir():
            yield cam


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_image(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def to_float_array(img: np.ndarray) -> np.ndarray:
    # Convert uint8 [0,255] to float32 [0,1]; keep float types as float32
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    return img.astype(np.float32)


def compute_diff(canonical: np.ndarray, target: np.ndarray) -> np.ndarray:
    # Ensure same size and channels
    if canonical.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: canonical {canonical.shape} vs target {target.shape}"
        )

    can_f = to_float_array(canonical)
    tar_f = to_float_array(target)
    # Signed difference: target - canonical (no absolute)
    diff = tar_f - can_f
    return diff


def process(
    root_dir: Path,
    canonical_dir: Path,
    out_root: Path,
    cam_start: int,
    cam_end: int,
    save_format: str = "npy",
    save_torch: bool = False,
):
    
    canonical_dir = root_dir / canonical_dir
    out_root = root_dir / out_root
    
    root_dir = root_dir.resolve()
    canonical_dir = canonical_dir.resolve()
    out_root =  out_root.resolve()

    ensure_dir(out_root)

    cams = list(list_cameras(root_dir, start=cam_start, end=cam_end))
    cam_iter = tqdm(cams, desc="Cameras") if tqdm is not None else cams
    for cam in cam_iter:
        cam_images_dir = root_dir / "images" / cam
        # Canonical file expected at meta/canonical_0/camXX.jpg
        canonical_path = canonical_dir / f"{cam}.jpg"
        if not canonical_path.exists():
            print(f"[WARN] Missing canonical image: {canonical_path}; skipping {cam}")
            continue

        try:
            canonical_img = read_image(canonical_path)
        except Exception as e:
            print(f"[WARN] Failed to read canonical {canonical_path}: {e}; skipping {cam}")
            continue

        # Walk all PNGs under images/camXX/* (e.g., 001.png, possibly nested)
        pngs = sorted(cam_images_dir.rglob("*.jpg"))
        if not pngs:
            print(f"[INFO] No JPG images for {cam} in {cam_images_dir}")
            continue

        img_iter = tqdm(pngs, desc=f"{cam}", leave=False) if tqdm is not None else pngs
        for png_path in img_iter:
            # Output should mirror under differences/camXX/<same relative path>.png
            rel = png_path.relative_to(cam_images_dir)
            # Change extension to .npy (or .pt if torch)
            if save_torch:
                out_fname = rel.with_suffix(".pt")
            else:
                out_fname = rel.with_suffix(".npy")
            out_path = out_root / cam / out_fname
            ensure_dir(out_path.parent)

            try:
                target_img = read_image(png_path)
                # Resize canonical to target if shapes differ but same channels
                diff = compute_diff(canonical_img, target_img)

                # Save as numpy .npy or torch .pt
                if save_torch:
                    if torch is None:
                        raise RuntimeError("Torch not available but --torch was requested")
                    # Save as CHW float32 tensor for typical DL workflows
                    if diff.ndim == 3:
                        # HWC -> CHW
                        tensor = torch.from_numpy(diff.transpose(2, 0, 1))
                    else:
                        tensor = torch.from_numpy(diff)
                    torch.save(tensor, str(out_path))
                else:
                    # Save float32 numpy array; keep HWC or HW as is
                    np.save(str(out_path), diff.astype(np.float32))
            except Exception as e:
                print(f"[ERROR] {cam}: failed on {png_path} -> {out_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute signed float per-pixel differences (target - canonical) between "
            "meta/canonical_0/camXX.jpg and images/camXX/*.png. Save as .npy (or .pt)."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/media/barry/56EA40DEEA40BBCD/DATA/studio_test3/"),
        help="Project root directory (default: current dir)",
    )
    parser.add_argument(
        "--canonical",
        type=Path,
        default=Path("meta/canonical_0"),
        help="Directory containing camXX.jpg canonical images",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("differences"),
        help="Output root directory for difference tensors",
    )
    parser.add_argument(
        "--cam-start",
        type=int,
        default=0,
        help="First camera index (inclusive), default 0",
    )
    parser.add_argument(
        "--cam-end",
        type=int,
        default=15,
        help="Last camera index (inclusive), default 15",
    )
    parser.add_argument(
        "--torch",
        action="store_true",
        help="Save tensors as Torch .pt (CHW) instead of NumPy .npy",
    )

    args = parser.parse_args()
    process(
        args.root,
        args.canonical,
        args.out,
        args.cam_start,
        args.cam_end,
        save_format="pt" if args.torch else "npy",
        save_torch=args.torch,
    )


if __name__ == "__main__":
    main()
