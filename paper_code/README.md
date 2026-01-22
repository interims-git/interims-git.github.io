
# Installation and Run

This has been tested on NVidia RTX 3090 w/ py3.10 pt2.4 and cu11.8, and NVidia RTX 4090 w/ py3.10 pt2.4 and cu12.4

1. Create a conda environment with `conda create -n vsres python=3.10` (pytorch version requires compatibility with `gsplat`; I use py310)
2. Install pytorch (requires compatibility with `gsplat`; I use pt24)
3. Download `gsplat` either via `pip install gsplat` or with wheel (I used `gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl`)
4. Run `pip install -r requirements.txt`
5. Use `bash run.sh SRC NAME ARGS` to run the method

## Texture Stats

For `texture_statistic.py` install
```
python3 -m venv texenv
source texenv/bin/activate
pip install numpy scikit-image matplotlib opencv-python
pip install joblib
```


# Data Collection Notes
See DATA_Instructions.md



