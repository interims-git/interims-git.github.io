Data Preprocessing Insturction

# 1. Video to Images
Going from video files to images. In this section "camXX.mp4" refers to the static (relighting) cameras and "camYY.mp4" refers to the canonical moving camera. CamYY is useful for pose estimation and generating the initial canonical representation. 

1. Extract all frames from all cameras and place into folders with the same name
2. For camXX, copy a single image/frame per background into a `meta/images/camXX`. Do not copy the canonical frame from these videos.
3. In each folder, rename the frames to ZZZ.jpg where ZZZ is a 3 digit number from 000 to the number of background processed
4. For camYY, copy viable frames into `pose_data/initial_images/`. Viable frames are not blurry (if you used an ISO) and are in focus.
5. Rename these 000.jpg to ZZZ.jpg and note down which 3digit number ZZZ is for the posing data (this will useful later). Lets call this number `MAX_POSE`.
6. For camXX, copy the canonical frame to `pose_data/initial_images/` and make sure to name it `{ZZZ+XX}.jpg`, where ZZZ+XX is the integer sum of the `MAX_POSE` and the relighting camera id. The ordering is important as we want images 000 to ZZZ to reference the pose-only data, and ZZZ+1 to ZZZ+XX to reference the poses for the relighting cameras.

# 2. Images to Dataset
Using Nerfstudio, pose the image data in `pose_data/initial_images/`. The command follows. If you have issues please consult the [Nerfstudio docs or github](https://docs.nerf.studio/quickstart/custom_dataset.html)

```
ns-process-data images --data {root}/pose_data/initial_images/ --output-dir {root}/pose_data/
```

# 3. Training the Splatfacto Model
Using Nerfstudio, train the splatfacto model using default setting. You can modify these yourself with the help of the `-h` CLI input. For example, we wanted to train full resolution so we set the downsample input to 1.

```
ns-train splatfacto --data {root}/pose_data/ -h
```

# 4. Exporting camera parameters and splat model
There are two commands we use follow. This should generate a `splat.ply` and a `transforms_train.json` files in the newly created `exports/` directory.
```
ns-export cameras --load-config {output/.../config.yml} --output-dir exports/

ns-export gaussian-splat --load-config {output/.../config.yml} --output-dir exports/

```

Copy `{nerfstudio}/exports/splat.ply` to `splat/splat.ply`.

Then open `pose_data/transforms.json` and replace the `transform_matrix` lists for each frame with the corresponding `transform_matrix` list in `{nerfstudio}/exports/transforms_train.ply`. If you want to make your life easier, you can ask chatGPT to do this task for you. It should produce a `pose_data/transforms.json` file that looks like:

```
{
    "w": 1920,
    "h": 1080,
    "fl_x": 1769.6365995009273,
    "fl_y": 1756.6433450750721,
    "cx": 958.5092506243253,
    "cy": 556.0282431322837,
    "k1": 0.019449745418577098,
    "k2": 0.08417273397351148,
    "p1": -0.0005665646315745907,
    "p2": -0.0004691999668991695,
    "camera_model": "OPENCV",

    "frames": [
        {
            "file_path": "images/frame_00180.jpg",
            "transform_matrix": [
                [
                    0.6894519925117493,
                    -0.034012049436569214,
                    0.7235323786735535,
                    0.7852264046669006
                ],
                [
                    0.7085449695587158,
                    -0.17574086785316467,
                    -0.68343186378479,
                    -0.05070751532912254
                ],
                [
                    0.1503991335630417,
                    0.9838487505912781,
                    -0.09706583619117737,
                    -0.177516907453537
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    1.0
                ]
            ],
            "colmap_im_id": 180
        },

        ...
```

Its important that `colmap_im_id` corresponds to `frame_XXX.jpg` as we want the last all cameras betwee XXX-`MAX_POSE` and XXX to correspond to the canonical frames for camXX, with the same order.

*Note that during dataloading we load the transforms for the relighting cams by sampling indexs* `i for i in range(len(transformsJSON["frames"])) if i > (len(transformsJSON["frames"]) - N)`, where N is the number of relighting cameras (training and testing cameras)

# 5. Final Notes