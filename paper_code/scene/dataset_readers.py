import os
from PIL import Image

from typing import NamedTuple

from utils.graphics_utils import getWorld2View2
import numpy as np
import torch
import json

import os
import json
import numpy as np
class CameraInfo(NamedTuple):
    R: np.array
    T: np.array
    fx: np.array
    fy: np.array
    cx: np.array
    cy: np.array
    
    k1: np.array
    k2: np.array
    p1: np.array
    p2: np.array

    image_path: str
    canon_path:str
    so_path: str
    
    image: torch.Tensor
    canon:torch.Tensor
    mask: torch.Tensor


    uid: int    
    width: int
    height: int
    time : int
    
    def update_canon(self, path):
        return self._replace(canon_path=path)
   
class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras:list
    video_cameras: list
    ba_background_fp:str
    nerf_normalization: dict
    background_pth_ids:list
    param_path:str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal #* 1.1

    translate = -center
    # breakpoint()
    return {"translate": translate, "radius": radius}

def readCamerasFromTransforms(path, transformsfile, plot=False):
    cam_infos = []

    tf_path = os.path.join(path, transformsfile)
    with open(tf_path, "r") as json_file:
        contents = json.load(json_file)

    # Global intrinsics
    g_fx = contents.get("fl_x")
    g_fy = contents.get("fl_y")
    g_cx = contents.get("cx")
    g_cy = contents.get("cy")
    g_w  = contents.get("w")
    g_h  = contents.get("h")
    g_k1 = contents.get("k1")
    g_k2 = contents.get("k2")
    g_p1 = contents.get("p1")
    g_p2 = contents.get("p2")

    # Nerfstudio normalization (transform + scale)
    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        fx = frame.get("fl_x", g_fx)
        fy = frame.get("fl_y", g_fy)
        cx = frame.get("cx", g_cx)
        cy = frame.get("cy", g_cy)
        w  = frame.get("w", g_w)
        h  = frame.get("h", g_h)

        k1 = frame.get("k1", g_k1)
        k2 = frame.get("k2", g_k2)
        p1 = frame.get("p1", g_p1)
        p2 = frame.get("p2", g_p2)

        # Load and convert transform
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        R =  c2w[:3, :3]
        T = c2w[:3, 3]
        
        image_path = os.path.normpath(os.path.join(path, frame["file_path"]))

        cam_infos.append(CameraInfo(
            uid=frame.get("colmap_im_id", idx),
            R=R, T=T,
            fx=fx, fy=fy, cx=cx, cy=cy,
            k1=k1, k2=k2, p1=p1, p2=p2,
            width=w, height=h,
            image_path=image_path,
            canon_path=None,
            so_path=None,
            image=None,
            canon=None,
            mask=None,
            time=float(frame.get("time", -1.0)),
        ))
    cam_infos.sort(key=lambda c: os.path.basename(c.image_path))

    if plot:
        import plotly.graph_objects as go
        from plyfile import PlyData

        plydata = PlyData.read(os.path.join(path, 'splat', 'splat.ply'))

        xyz = np.stack((
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ), axis=1)

        fig = go.Figure()

        # --- Add point cloud ---
        fig.add_trace(go.Scatter3d(
            x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
            mode='markers',
            marker=dict(size=1, opacity=0.3, color='gray'),
            name='Point Cloud'
        ))

        axis_len = 0.1
        N = len(cam_infos) - 100

        # Helper to add an arrow using a line segment
        def add_arrow(fig, start, vec, color, name=None):
            fig.add_trace(go.Scatter3d(
                x=[start[0], start[0] + vec[0]],
                y=[start[1], start[1] + vec[1]],
                z=[start[2], start[2] + vec[2]],
                mode='lines',
                line=dict(width=6, color=color),
                name=name,
                showlegend=False
            ))

        for id, cam in enumerate(cam_infos):
            T = cam.T
            R = cam.R

            # --- Camera center ---
            fig.add_trace(go.Scatter3d(
                x=[T[0]], y=[T[1]], z=[T[2]],
                mode='markers+text',
                marker=dict(size=4, color='black'),
                text=[str(id)],
                textposition='top center',
                name=f'cam {id}',
                showlegend=False
            ))

            # Axes
            x_axis = R[:, 0] * axis_len
            y_axis = R[:, 1] * axis_len
            z_axis = R[:, 2] * axis_len
            forward = -R[:, 2] * 0.2

            add_arrow(fig, T, x_axis, "red")
            add_arrow(fig, T, y_axis, "green")
            add_arrow(fig, T, z_axis, "blue")
            add_arrow(fig, T, forward, "cyan")

        # Layout
        fig.update_layout(
            title="Camera Centers + Rotation Directions (Plotly)",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
            ),
            width=1000,
            height=800,
        )

        fig.show()
        exit()
    return cam_infos

from torchvision import transforms as T
TRANSFORM = T.ToTensor()

def readCamerasFromCanon(path, canon_cams, M=19, preload_gpu=False, subset=1):
    background_path = os.path.join(path, 'meta', 'backgrounds')
    background_im_paths = [os.path.join(background_path, f) for f in sorted(os.listdir(background_path))]
    relit_path = os.path.join(path, 'meta', 'images')
    masks_path = os.path.join(path, 'meta', 'masks')

    # Get the colmap id for the first relit camera (i.e. the last 19 frames of the nerfstudio dataset)
    N = len(canon_cams) - M

    image=None
    canon=None
    mask=None

    # Set the subset flags to only store info on one of the three subsets
    if subset == 1:
        min_b_id = 0
        max_b_id = 33
    elif subset == 2:
        min_b_id = 33
        max_b_id = 66
    elif subset == 3:
        min_b_id = 66
        max_b_id = 99
    elif subset == -1:
        min_b_id = 0
        max_b_id = 99
        
    relit_cams = []
    background_im_paths_ = []
    store_bck = True
    for cam in canon_cams:
        if cam.uid > N:
            cam_id = cam.uid - N
            cam_name = f'cam{cam_id:02}'

            for background_id, b_path in enumerate(background_im_paths):
                if background_id >= min_b_id and background_id < max_b_id:
                    if store_bck == True: # store only relevant background paths
                        background_im_paths_.append(b_path)
                    
                    im_name = b_path.split('/')[-1].replace('png', 'jpg') # e.g. '000.jpg'
                    im_path = os.path.join(relit_path, cam_name, im_name)
                    mask_path = os.path.join(masks_path, f'{cam_name}.png')
                    
                    # Load 
                    if preload_gpu:
                        img = Image.open(im_path).convert("RGB")
                        img = img.resize(
                            (cam.width, cam.height),
                            resample=Image.LANCZOS  # or Image.NEAREST, Image.BICUBIC, Image.LANCZOS
                        )            
                        image = TRANSFORM(img).cuda()
                        
                        img = Image.open(cam.image_path).convert("RGB")
                        img = img.resize(
                            (cam.width, cam.height),
                            resample=Image.LANCZOS  # or Image.NEAREST, Image.BICUBIC, Image.LANCZOS
                        )            
                        canon = TRANSFORM(img).cuda()
                        img = Image.open(mask_path).split()[-1]
                        img = img.resize(
                            (cam.width, cam.height),
                            resample=Image.LANCZOS  # or Image.NEAREST, Image.BICUBIC, Image.LANCZOS
                        )

                        mask = 1. - TRANSFORM(img).cuda()

                    time = background_id - min_b_id
                    cam_info = CameraInfo(
                        uid=cam.uid, 
                        R=cam.R, T=cam.T,
                        
                        fx=cam.fx,
                        fy=cam.fy,
                        cx=cam.cx, cy=cam.cy,
                        k1=cam.k1, k2=cam.k2, p1=cam.p1, p2=cam.p2,

                        width=cam.width, height=cam.height,

                        image_path=im_path, 
                        canon_path=cam.image_path,
                        so_path=mask_path,
                        
                        image=image,
                        canon=canon,
                        mask=mask,
                        
                        time = time,
                    )
                    relit_cams.append(cam_info)
            store_bck = False
    return relit_cams, background_im_paths_

def generate_circular_cams(
    path,
    cam,
    idx_targets=None
):
    """
    Generate a circular camera path (all c2w) around a fixed point in front
    of the input camera, keeping orientation toward that point.
    Assumes OpenCV-style camera with forward = -Z in c2w.
    """

    # --- extract camera basis (columns of R are world-space camera axes)
    fp = os.path.join(path, 'meta/video_paths.json')
    try:
        with open(fp, "r") as json_file:
            contents = json.load(json_file)

        zoomscale = 1.
        # --- recompute fx, fy from that new FOV
        fx_new = cam.fx*zoomscale
        fy_new = cam.fy*zoomscale
        cams=[]
        if idx_targets is not None:
            total_frame_length = len(contents['camera_path'])
            total_background_files = len(idx_targets)
            factor = int(total_frame_length/total_background_files) # floor the number of frames per background
            
            if factor < 1: # when total_frame_length < total_background_files
                background_idxs = [idx_targets[i] for i in range(total_frame_length)]
            else:
                background_idxs = []
                for i in range(total_background_files):
                    for j in range(factor):
                        background_idxs.append(i)
                
                # Now fill out any spacing left by flooring the factor
                background_idxs = background_idxs + [idx_targets[-1] for i in range(total_frame_length - len(background_idxs))]
        else:
            background_idxs = [idx_targets[0] for i in range(len(contents['camera_path']))]
            
        for idx, info in enumerate(contents['camera_path']):
            c2w = np.array(info["camera_to_world"], dtype=np.float32).reshape(4, 4)

            R = c2w[:3, :3]
            T = c2w[:3, 3]

            cam_info = CameraInfo(
                        uid=cam.uid, 
                        R=R, T=T,
                        
                        fx=fx_new,
                        fy=fy_new,
                        cx=cam.cx, cy=cam.cy,
                        k1=cam.k1, k2=cam.k2, p1=cam.p1, p2=cam.p2,

                        width=cam.width, height=cam.height,
                        image=None,
                        canon=None,
                        mask=None,
                        image_path=None, 
                        canon_path=None,
                        so_path=None,
                        time=background_idxs[idx],
                    )
            cams.append(cam_info)
    except:
        cams = None
    return cams


def readSceneInfo(path, preload_imgs=False, additional_dataset_args=1, N_test_frames=10, cam_config=1, canon_args=None):
    """Construct dataset from nerfstudio
    
    Args:
        path: str, path to dataself folder (containnig images/meta/transforms.json/splat)
        preload_imgs: bool, to load images to gpu before training (default: false)
        additional_dataset_args: int, The scene configuration to use (1-3 for Scenes 1-3) (main results in paper)
        N_test_frames: int, number of IBL sources to use for testing. The remaining IBL backgrounds are used for training (anlations in paper)
        can_config: int, if in [6,12] choose the 6-camera or 12-camera scene configurations (ablations in paper)
        canon_args: dict, {"canon_data": "lit" if we use a lit reference as canon or "unlit" for unlit reference, ...}
    """
    print(f"Reading {path.split('/')[-1]} & subset {additional_dataset_args} ...")
    assert additional_dataset_args in [1,2,3, -1, -2], f"--subset needs to be [1,2,3, -1]"
    assert N_test_frames >-1, f"--test-frames needs to be > -1"
    
    # Read camera transforms    
    canon_cam_infos = readCamerasFromTransforms(path, 'transforms.json')
    
    if path.split('/')[-1] == 'scene3' and cam_config in [6, 12]:
        if cam_config == 6:
            target_infos = [2,3,6,11,12,17,18]
            V_cam = 6
        elif cam_config == 12:
            target_infos = [6,7,8,9,10,11,12,13,14,15,16,17,18]
            V_cam = 12
            
        new_cam_infos = []
        for idx, cam in enumerate(canon_cam_infos):
            if idx in target_infos:
                new_cam_infos.append(cam)
                
        canon_cam_infos = new_cam_infos
        M = len(target_infos)
        
    else:
        if 'scene1' in path:
            V_cam = 5
        else:
            V_cam = 18
            
        M = 19
        
    # This should return 18x33=627 CameraInfo classes
    cam_infos, background_paths = readCamerasFromCanon(path, canon_cam_infos, M=M, preload_gpu=preload_imgs, subset=additional_dataset_args)  # L should be the number of background paths
    
    L = len(background_paths)
    
    if canon_args["single_frame"] != -1:
        print(f"- overriding training for a single frame for {canon_args['single_frame']}")
        assert canon_args["single_frame"] > -1 and canon_args["single_frame"] < L, f"--train-single-frame input need to be between 0 and {L}"
        N_test_frames = L-1
        L_test_idx_set = [i for i in range(L) if i != canon_args["single_frame"]] # The lighting-only test set (the first 10 frames for each camera)

    else:
        assert N_test_frames < L, f"--test-frames needs to be < {L} (the # of background textures)"
        print(f" - using 0-{N_test_frames} for testing leaving {L- N_test_frames} for training")
        print(f" - using camera {V_cam} for testing novel view synthesis")
    
        L_test_idx_set = [i for i in range(N_test_frames)] # The lighting-only test set (the first 10 frames for each camera)
        
    V_test_idx_set = [(V_cam*L)+i for i in range(L) if i not in L_test_idx_set] # The novel-view only test set
    LV_test_idx_set = [(V_cam*L)+i for i in range(L) if i in L_test_idx_set] # The novel-view & novel lighting test set

    # Load the split datasets
    L_test_cams = [cam for idx, cam in enumerate(cam_infos)  if (idx % L) in L_test_idx_set and idx not in LV_test_idx_set] # For indexs n the lighting test set
    V_test_cams = [cam for idx, cam in enumerate(cam_infos) if idx in V_test_idx_set] # For indexs in the novel view test set
    LV_test_cams = [cam for idx, cam in enumerate(cam_infos) if idx in LV_test_idx_set] # For indexs in the novel view and novel lighting test set
    test_cams = [L_test_cams, V_test_cams, LV_test_cams]

    # We should have 18x23=414 training images for the scene for the vanilla tests
    relighting_cams = [cam for idx, cam in enumerate(cam_infos) if (idx % L) not in L_test_idx_set and idx not in V_test_idx_set] # For indexs not in lighting and novel view cameras

    # Select cameras with a common background for pose estimation (from the training set)
    selected_background_fp = background_paths[0]
    
    # Camera path for novel view videos
    video_cams = generate_circular_cams(path, cam_infos[V_cam], idx_targets=L_test_idx_set)
    if video_cams is None: # TODO: Add script for video paths for this scene
        print("No video cams defaulting to View only test data")
        video_cams = test_cams[1]
        
        
    # If we want to train from a lit reference:
    print(f"- using {canon_args['canon_data']} images for training canon")
    if canon_args["canon_data"] == "lit" and preload_imgs == False:
        # `relighting_cams` order w.r.t (j,k): [(0,0), (0,1), ..., (0, K), (1,0), ..., (J,K)]
        K = L- N_test_frames
        J = M-1
        assert J*K == len(relighting_cams), f"the predicted J={J} x K={K} != {len(relighting_cams)} number of relighting cams in list"

        new_relighting_set = []
        for j in range(J):
            new_canon_fp = relighting_cams[j*K].image_path
            for k in range(1, K): # skip the first one
                relighting_cams[j*K+k] = relighting_cams[j*K+k].update_canon(new_canon_fp)
                new_relighting_set.append(relighting_cams[j*K+k])

        relighting_cams = new_relighting_set
        print(f"... now using {J*(K-1)} trainging images and using k=0 for canonical training")    
        
    nerf_normalization = getNerfppNorm(relighting_cams)

    scene_info = SceneInfo(
        train_cameras=relighting_cams,
        test_cameras=test_cams,
        video_cameras=video_cams,
        
        ba_background_fp=selected_background_fp,
        
        nerf_normalization=nerf_normalization,
        background_pth_ids=background_paths,

        param_path=os.path.join(path, 'splat','splat.ply') # loaded in gaussian_model.load_ply

    )
    return scene_info


sceneLoadTypeCallbacks = {
    "vsr":readSceneInfo,
}
