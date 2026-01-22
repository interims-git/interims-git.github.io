import os
import torch
import lpips
from PIL import Image
import torchvision.transforms as T
import warnings
import contextlib
import sys
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="torchvision"
)

# -------------------------
# 2. Suppress LPIPS print output
# -------------------------
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as f:
        old_stdout = sys.stdout
        sys.stdout = f
        try:
            yield
        finally:
            sys.stdout = old_stdout
            
            
dataset_num = '3'
scene_num = '3'
scene_name = 'C2'
datatype = 'LV'

# ------------------
# Config
# ------------------
gen_dir = f"./output/studio_test5/scene{dataset_num}/{scene_name}_{dataset_num}.{scene_num}/tests"
gt_root = f"./output/test_views/s{dataset_num}/{scene_num}/"
gt_dir = gt_root + f"imgs"
mask_dir =  gt_root + f"masks"


device = "cuda" if torch.cuda.is_available() else "cpu"
with suppress_stdout():
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
lpips_fn.eval()

# Image transforms
to_tensor = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,)*3, (0.5,)*3)        # [-1,1]
])

mask_transform = T.ToTensor()  # mask stays in [0,1]

# ------------------
# Helper
# ------------------
def load_image(path):
    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

def load_mask(path):
    mask = Image.open(path).convert("L")
    mask = mask_transform(mask).unsqueeze(0).to(device)
    return mask

# ------------------
# Main loop
# ------------------
lpips_scores = []

for fname in os.listdir(gen_dir):
    if not fname.endswith(".jpg"):
        continue

    idx = fname.split("_")[-1].replace(".jpg", "")  # extract 00000
    if datatype in fname:

        gen_path = os.path.join(gen_dir, fname)
        gt_path = os.path.join(gt_dir, f"{idx}.jpg")
        mask_path = os.path.join(mask_dir, f"mask{idx}.jpg")

        if not (os.path.exists(gt_path) and os.path.exists(mask_path)):
            continue

        gen_img = load_image(gen_path)
        gt_img = load_image(gt_path)
        mask = load_mask(mask_path)

        # Apply mask
        gen_masked = gen_img #* mask
        gt_masked = gt_img #* mask

        with torch.no_grad():
            score = lpips_fn(gen_masked, gt_masked)
            lpips_scores.append(score.item())

# ------------------
# Result
# ------------------
avg_lpips = sum(lpips_scores) / len(lpips_scores)
print(f"{avg_lpips}")
