import json
import glob
import re
import matplotlib.pyplot as plt
import argparse
import os

def plot_all_keys(folder: str):
    keys = ["L", "LV", "V"]
    metrics_to_plot = ["psnr", "psnr-y", "psnr-crcb", "ssim"]

    # Validate folder
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Get all JSON files
    files = sorted(
        glob.glob(os.path.join(folder, "metrics_*.json")),
        key=lambda x: int(re.findall(r"(\d+)", os.path.basename(x))[0])
    )

    if not files:
        raise FileNotFoundError(f"No metrics_*.json files found in {folder}")

    # Extract numeric steps from file basenames
    steps = [int(re.findall(r"(\d+)", os.path.basename(f))[0]) for f in files]

    # Initialize data containers
    data_all = {k: {m: [] for m in metrics_to_plot} for k in keys}

    # Load all data
    for f in files:
        with open(f, "r") as fh:
            js = json.load(fh)
        for k in keys:
            if k not in js:
                raise KeyError(f"Key '{k}' not found in {f}")
            for m in metrics_to_plot:
                data_all[k][m].append(js[k][m])

    # === Plot ===
    fig, axes = plt.subplots(len(metrics_to_plot), len(keys), figsize=(12, 10), sharex=True)
    fig.suptitle(f"Metrics from folder: {os.path.abspath(folder)}", fontsize=14, weight="bold")

    for row, metric in enumerate(metrics_to_plot):
        for col, key in enumerate(keys):
            ax = axes[row, col]
            ax.plot(steps, data_all[key][metric], marker="o")
            ax.grid(True)
            if row == 0:
                ax.set_title(f"{key}", fontsize=12, weight="bold")
            if col == 0:
                ax.set_ylabel(metric, fontsize=10)
            if row == len(metrics_to_plot) - 1:
                ax.set_xlabel("Step")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot PSNR and SSIM metrics for all keys (L, LV, V).")
    parser.add_argument("--folder", type=str, required=False, default=".", help="Folder containing metrics_*.json files")

    args = parser.parse_args()
    plot_all_keys(args.folder)
