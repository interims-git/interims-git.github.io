import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from joblib import Parallel, delayed
import glob
import argparse
import os
from collections import defaultdict

# -------------------------------------------
# PARAMETERS
# -------------------------------------------
frequencies = [0.1, 0.2, 0.3]
thetas = [0, np.pi/4, np.pi/2]
KERNEL_CACHE = {}

# -------------------------------------------
# FAST GABOR
# -------------------------------------------
def get_gabor_kernel(freq, theta, ksize=31, sigma=4.0):
    key = (freq, theta)
    if key not in KERNEL_CACHE:
        lambd = 1.0 / freq
        kernel = cv2.getGaborKernel(
            (ksize, ksize),
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=0.5,
            psi=0,
            ktype=cv2.CV_32F
        )
        KERNEL_CACHE[key] = kernel
    return KERNEL_CACHE[key]


def fast_gabor_features_rgb(img_rgb, freqs, thetas):
    mags_rgb = []
    for f in freqs:
        for t in thetas:
            kernel = get_gabor_kernel(f, t)

            # Split channels
            r = img_rgb[:, :, 0]
            g = img_rgb[:, :, 1]
            b = img_rgb[:, :, 2]

            # Filter each channel
            real_r = cv2.filter2D(r, cv2.CV_32F, kernel)
            real_g = cv2.filter2D(g, cv2.CV_32F, kernel)
            real_b = cv2.filter2D(b, cv2.CV_32F, kernel)

            # Magnitude
            mag_r = np.abs(real_r)
            mag_g = np.abs(real_g)
            mag_b = np.abs(real_b)

            mags_rgb.append((mag_r, mag_g, mag_b))
    return mags_rgb


def gabor_frequency_score_rgb(gabor_rgb):
    energies = []
    for (mr, mg, mb) in gabor_rgb:
        e_r = np.mean(mr * mr)
        e_g = np.mean(mg * mg)
        e_b = np.mean(mb * mb)
        energies.append((e_r + e_g + e_b) / 3.0)
    return float(np.mean(energies))


# -------------------------------------------
# GLCM REGULARITY
# -------------------------------------------
def glcm_regularity(img):
    img8 = (img * 255).astype(np.uint8)
    distances = [1, 2, 4]
    angles   = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    glcm = graycomatrix(
        img8,
        distances=distances,
        angles=angles,
        symmetric=True,
        normed=True,
    )
    return graycoprops(glcm, 'homogeneity').mean()


# -------------------------------------------
# PROCESS ONE IMAGE WITH MASK
# -------------------------------------------
def process_image(path, mask_root):
    # Extract camera name (cam02)
    cam_name = os.path.basename(os.path.dirname(path))

    # Mask path
    mask_path = os.path.join(mask_root, f"{cam_name}.png")
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Missing mask: {mask_path}")

    # Load image (float normalized)
    img = cv2.imread(path).astype(np.float32) / 255.0
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize mask if needed
    if mask.shape != img_rgb.shape[:2]:
        mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    mask_bin = (mask > 0).astype(np.float32)

    # Apply mask to RGB
    masked_rgb = img_rgb * mask_bin[:, :, None]

    # Compute RGB Gabor features
    gabs_rgb = fast_gabor_features_rgb(masked_rgb, frequencies, thetas)
    freq_score = gabor_frequency_score_rgb(gabs_rgb)

    # GLCM still uses grayscale
    img_gray = rgb2gray(img_rgb).astype(np.float32)
    masked_gray = img_gray * mask_bin
    reg_score = glcm_regularity(masked_gray)

    frame_name = os.path.basename(path)
    return frame_name, freq_score, reg_score


# -------------------------------------------
# MAIN
# -------------------------------------------
def main(data_root):

    image_root = os.path.join(data_root, "images")
    mask_root  = os.path.join(data_root, "masks")

    image_paths = sorted(glob.glob(os.path.join(image_root, "cam*", "*.jpg")))
    print(f"Found {len(image_paths)} images.")

    results = Parallel(n_jobs=-1)(
        delayed(process_image)(p, mask_root) for p in image_paths
    )

    # Group metrics by frame name (e.g. 001.jpg)
    grouped_freq = defaultdict(list)
    grouped_reg  = defaultdict(list)

    for frame_name, freq, reg in results:
        grouped_freq[frame_name].append(freq)
        grouped_reg[frame_name].append(reg)

    # Compute averages per frame across all cameras
    avg_freq_per_frame = {}
    avg_reg_per_frame  = {}

    for frame_name in sorted(grouped_freq.keys()):
        avg_freq_per_frame[frame_name] = np.mean(grouped_freq[frame_name])
        avg_reg_per_frame[frame_name]  = np.mean(grouped_reg[frame_name])

    frames = sorted(avg_freq_per_frame.keys())
    freq_vals = [avg_freq_per_frame[f] for f in frames]
    reg_vals  = [avg_reg_per_frame[f] for f in frames]

    # Print summary
    print("\nAVERAGES PER FRAME (masked):")
    for f in frames:
        print(f"{f}: freq={avg_freq_per_frame[f]:.6f}, reg={avg_reg_per_frame[f]:.6f}")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(freq_vals, reg_vals, s=70)
    for i, f in enumerate(frames):
        plt.annotate(f.split('.')[0], (freq_vals[i], reg_vals[i]))

    plt.xlabel("Average Frequency (masked)")
    plt.ylabel("Average Regularity (masked)")
    plt.title("Per-frame texture metrics (masked regions removed)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    main(args.data_path)
