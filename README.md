# Code Installation and Run

This has been tested on Nvidia RTX 3090 w/ py3.10 pt2.4 and cu11.8, Nvidia RTX 4090 w/ py3.10 pt2.4 and cu12.4 and NVidia H100 w/ py3.10 pt2.4 and cu12.4

1. Create a conda environment with `conda create -n vsres python=3.10` (pytorch version requires compatibility with `gsplat`; I use py310)
2. Install pytorch (requires compatibility with `gsplat`; I use pt24)
3. Download `gsplat` either via `pip install gsplat` or with wheel (I used `gsplat-1.5.3+pt24cu124-cp310-cp310-linux_x86_64.whl`)
4. Run `pip install -r requirements.txt`
5. Use `bash run.sh SRC NAME ARGS` to run the method

`SRC` should be the file path to the data folder. This should end with e.g. ".../scene1" or ".../scene2" or ".../scene3".

`Name` is the experiment name

For training `ARGS` should be the subset we want to train. Choose from `[1, 2, 3]`.

For viewing (after training) `ARGS= "view [checkpoint iteration]"`, where `[checkpoint iteration]` should be replaced by the checkpoint iteration you want to view/load.


# Training Viewer
- View the scene from the training and testing views
- Change the render flags to view depth, XYZ, or any of the other AOVs
- Provides active testing and rendering stats
- Drag the slider to change the background image
- Switch on/off viewing the final relit scene - Note that if this is activated then the other render flags will not work

# Post-Training Viewer
- To pose the background mesh, hit the button and drag the corners of the input texture 
- Insert a path to a background video and play/save the resulting render
- Save the current frame

# Data Collection Notes
See paper_code/DATA_Instructions.md


