try:
    import dearpygui.dearpygui as dpg
except:
    print("No dpg running")
    dpg = None

import numpy as np
import os
import copy
import psutil
import torch
from gaussian_renderer import render, render_ibl_pose_points
from tqdm import tqdm
import time
import json
import cv2
from torchvision import transforms
import threading
import time
to_tensor = transforms.ToTensor()  # auto converts HWC uint8 → CHW float32 in [0,1]

class GUIBase:
    """This method servers to intialize the DPG visualization (keeping my code cleeeean!)
    
        Notes:
            none yet...
    """
    def __init__(self, gui, scene, gaussians, runname, view_test):
        
        self.gui = gui
        self.scene = scene
        self.gaussians = gaussians
        self.runname = runname
        self.view_test = view_test
        
        # Set the width and height of the expected image
        self.W, self.H = self.scene.train_camera[0].image_width, self.scene.train_camera[0].image_height

        if self.H > 1200 and self.scene != "dynerf":
            self.W = self.W//2
            self.H = self.H //2
        # Initialize the image buffer
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        
        # Other important visualization parameters
        self.time = 0
        self.show_radius = 30.
        self.vis_mode = 'render'
        self.show_mask = 'none'
        self.show_dynamic = 0.
        self.w_thresh = 0.
        self.h_thresh = 0.
        
        self.set_w_flag = False
        self.w_val = 0.01
        # Set-up the camera for visualization
        self.show_scene_target = 0

        self.finecoarse_flag = True        
        self.switch_off_viewer = False
        self.switch_off_viewer_args = False
        self.full_opacity = False
        
        # Rendering/Novel View Settings
        self.novel_view_background_dir = ""
        self.drag_func = "viewing"
        self.drag_im_buffer = None
        self.trainable_abc = None
        
        # Analysis/Inspection tools
        self.mous_loc = [0, 0] # x,y
        self.mous_loc_last = [0, 0] # x,y

        # Viewer settings for camera/view selection
        self.free_cams = [cam for idx, cam in enumerate(self.scene.test_camera) if idx % self.N_test_frames == 0] 
        self.current_cam_index = 0
        self.original_cams = [copy.deepcopy(cam) for cam in self.free_cams]
        self.play_custom_video = False
        self.save_custom_video = False
        self.save_frame=False
        
        if self.gui:
            print('DPG loading ...')
            dpg.create_context()
            self.register_dpg()
            

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def track_cpu_gpu_usage(self, time):
        # Print GPU and CPU memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 ** 2)  # Convert to MB

        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
        print(
            f'[{self.stage} {self.iteration}] Time: {time:.2f} | Allocated Memory: {allocated:.2f} MB, Reserved Memory: {reserved:.2f} MB | CPU Memory Usage: {memory_mb:.2f} MB')
    
    def render(self):
        cnt = 0
        if self.gui:
            while dpg.is_dearpygui_running():
                if self.view_test == False:


                    dpg.set_value("_log_stage", self.stage)

                    if self.iteration <= self.final_iter:
                        # Get batch data

                        viewpoint_cams = self.get_batch_views
                        
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        self.iter_start.record()
                        
                        # Depending on stage process a training step
                        self.train_step(viewpoint_cams)
                    
                        self.iter_end.record()
                    
                    else: # Initialize fine from coarse stage
                        if self.stage == 'fine':
                            self.stage = 'done'
                            view_size=len(self.scene.video_camera)
                            for i, test_cam in enumerate(self.scene.video_camera):
                                metric_results = self.video_step(test_cam, i)

                                dpg.set_value("_log_test_progress", f"{int(100*(i/view_size))}% | novel view video")
                                dpg.render_dearpygui_frame()

                            dpg.stop_dearpygui()
                            
                    # Test Step
                    if self.iteration % self.test_every == 0:
                        metrics = {
                            "mse":0.,
                            "psnr":0.,
                            "psnr-y":0.,
                            "psnr-crcb":0.,
                            "ssim":0.
                        }
                        
                        datasets = {
                            "L":metrics.copy(),
                            "V":metrics.copy(),
                            "LV":metrics.copy(),
                        }
                        test_size = len(self.scene.test_camera)
                        dataset_idxs = self.scene.test_camera.subset_idxs
                        cnt = 0
                        for i, test_cam in enumerate(self.scene.test_camera):
                            if dataset_idxs is not None:
                                if i < dataset_idxs[0]: # L-only tests
                                    d_type = "L"
                                elif i < dataset_idxs[0] + dataset_idxs[1]: # V-only test
                                    d_type = "V"
                                else:
                                    d_type = "LV"
                            else:
                                d_type = "LV"
                                cnt += 1
                            metric_results = self.test_step(test_cam, i, d_type)
                        
                                
                            for key in metrics.keys():
                                datasets[d_type][key] += metric_results[key].item()
                            
                            dpg.set_value("_log_test_progress", f"{int(100*(i/test_size))}% | {metric_results['psnr']:.2f} on {d_type} type")
                            dpg.render_dearpygui_frame()

                        # Average
                        if dataset_idxs is None:
                            dataset_idxs = [test_size]

                        for key, lengths in zip(datasets.keys(), dataset_idxs):
                            for key_1 in metrics.keys():
                                datasets[key][key_1] /= lengths

                            # Logs with 3 decimal places
                            dpg.set_value("_log_l_1", f"mse  : {datasets['L']['mse']:.3f}")
                            dpg.set_value("_log_l_2", f"ssim : {datasets['L']['ssim']:.3f}")
                            dpg.set_value("_log_l_3", f"psnr : {datasets['L']['psnr']:.2f}")
                            dpg.set_value("_log_l_4", f"psnr-y : {datasets['L']['psnr-y']:.2f}")
                            dpg.set_value("_log_l_5", f"psnr-crcb : {datasets['L']['psnr-crcb']:.2f}")

                            dpg.set_value("_log_v_1", f"mse  : {datasets['V']['mse']:.3f}")
                            dpg.set_value("_log_v_2", f"ssim : {datasets['V']['ssim']:.3f}")
                            dpg.set_value("_log_v_3", f"psnr : {datasets['V']['psnr']:.2f}")

                        dpg.set_value("_log_lv_1", f"mse  : {datasets['LV']['mse']:.3f}")
                        dpg.set_value("_log_lv_2", f"ssim : {datasets['LV']['ssim']:.3f}")
                        dpg.set_value("_log_lv_3", f"psnr : {datasets['LV']['psnr']:.2f}")
                        dpg.set_value("_log_lv_4", f"psnr-y : {datasets['LV']['psnr-y']:.2f}")
                        dpg.set_value("_log_lv_5", f"psnr-crcb : {datasets['LV']['psnr-crcb']:.2f}")
                        dpg.set_value("_log_test_progress", f"Saving json ...")
                        
                        test_fp = os.path.join(self.statistics_path, f"metrics_{self.iteration}.json")
                        with open(test_fp, "w") as outfile:
                            json.dump(datasets, outfile, indent=4, ensure_ascii=False)
                            
                            
                        dpg.set_value("_log_test_progress", f"...(training)...")
                        dpg.render_dearpygui_frame()
                        
                    # Update iteration
                    self.iteration += 1
                elif cnt ==0:
                    self.initialize_abc()
                    cnt = 1
                    
                    # view_size=len(self.scene.video_camera)
                    # for i, test_cam in enumerate(self.scene.video_camera):
                    #     metric_results = self.video_step(test_cam, i, abc=self.abc)
                    #     dpg.render_dearpygui_frame()


                with torch.no_grad():
                    if self.play_custom_video:
                        self.play_video()
                        self.switch_off_viewer = False
                          
                    elif self.switch_off_viewer == False:
                        self.viewer_step()
                        dpg.render_dearpygui_frame()   
                    else:
                        dpg.render_dearpygui_frame()   

                    
                with torch.no_grad():
                    self.timer.pause() # log and save
                    torch.cuda.synchronize()
                    if self.iteration % 1000 == 500: # make it 500 so that we dont run this while loading view-test
                        self.track_cpu_gpu_usage(0.1)
                        
                    # Save scene when at the saving iteration
                    if self.stage == 'fine' and (self.iteration == self.final_iter-1):
                        self.save_scene()

                    self.timer.start()
                    
            dpg.destroy_context()
        else:
            while self.stage != 'done':
                if self.view_test == False:
                    if self.iteration % 100 == 0:
                        print(f"[ITER {self.iteration}/{self.final_iter}] Training ...")
                    if self.iteration <= self.final_iter:
                        # Get batch data
                        viewpoint_cams = self.get_batch_views
                        
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        self.iter_start.record()
                        
                        # Depending on stage process a training step
                        self.train_step(viewpoint_cams)

                        self.iter_end.record()
                    
                    else: # Initialize fine from coarse stage
                        if self.stage == 'fine':
                            self.stage = 'done'
                            view_size=len(self.scene.video_camera)
                            for i, test_cam in enumerate(self.scene.video_camera):
                                metric_results = self.video_step(test_cam, i)
                            
                    # Test Step
                    if self.iteration % self.test_every == 0:
                        metrics = {
                            "mse":0.,
                            "psnr":0.,
                            "psnr-y":0.,
                            "psnr-crcb":0.,
                            "ssim":0.
                        }
                        
                        datasets = {
                            "L":metrics.copy(),
                            "V":metrics.copy(),
                            "LV":metrics.copy(),
                        }
                        test_size = len(self.scene.test_camera)
                        dataset_idxs = self.scene.test_camera.subset_idxs

                        for i, test_cam in enumerate(self.scene.test_camera):
                            if i < dataset_idxs[0]: # L-only tests
                                d_type = "L"
                            elif i < dataset_idxs[0] + dataset_idxs[1]: # V-only test
                                d_type = "V"
                            else:
                                d_type = "LV"
                                
                            metric_results = self.test_step(test_cam, i, d_type)
                        
                                
                            for key in metrics.keys():
                                datasets[d_type][key] += metric_results[key].item()
                            
                        # Average
                        for key, lengths in zip(datasets.keys(), dataset_idxs):
                            for key_1 in metrics.keys():
                                datasets[key][key_1] /= lengths

                        
                        test_fp = os.path.join(self.statistics_path, f"metrics_{self.iteration}.json")
                        with open(test_fp, "w") as outfile:
                            json.dump(datasets, outfile, indent=4, ensure_ascii=False)
                        
                    # Update iteration
                    self.iteration += 1
              
                with torch.no_grad():
                    self.timer.pause() # log and save
                    torch.cuda.synchronize()
                    if self.iteration % 1000 == 500: # make it 500 so that we dont run this while loading view-test
                        self.track_cpu_gpu_usage(0.1)
                        
                    # Save scene when at the saving iteration
                    if self.stage == 'fine' and (self.iteration == self.final_iter-1):
                        self.save_scene()

                    self.timer.start()
                     
    @torch.no_grad()
    def viewer_step(self):
        t0 = time.time()
        mous_hover_value = [0.]

        cam = self.free_cams[self.current_cam_index]
        cam.time = self.time
        
        try:
            abc = self.abc.abc
        except:
            abc = None
        id1 = self.time % len(self.scene.ibl)
        
        texture = self.scene.ibl[id1].cuda()
        
        buffer_image = render(
                cam,
                self.gaussians,
                abc,
                texture,
                view_args={
                    "vis_mode":self.vis_mode,
                    "stage":self.stage,
                    "finecoarse_flag":self.finecoarse_flag
                },
                mip_level=self.opt.mip_level
        )

        try:
            buffer_image = buffer_image["render"]
        except:
            print(f'Mode "{self.vis_mode}" does not work')
            buffer_image = buffer_image['render']
        
        # Render texture ABC vertices
        if self.drag_func == "ibl-pose":
            abc_rgba = render_ibl_pose_points(
                cam,
                abc,
                mip_level=self.opt.mip_level
            )
            self.drag_im_buffer = abc_rgba
            
            buffer_image = abc_rgba[-1, ...]*abc_rgba[:3, ...] +(1. - abc_rgba[-1, ...]) * buffer_image
        
        # Display value of image at current mouse position
        try:
            mous_hover_value = buffer_image[:, self.mous_loc[1], self.mous_loc[0]]
        except:
            mous_hover_value = [0.]
        
        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H,self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        
        try:
            if self.show_mask == 'occ':
                mask = cam.sceneoccluded_mask.squeeze(0).cuda()
            else:
                mask = 0.
            buffer_image[0] += mask*0.5
        except:
            pass
        
        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        t1 = time.time()
        buffer_image = self.buffer_image

        if self.save_frame:
            frame_uint8 = (buffer_image * 255).astype("uint8")
            frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite("current_frame.png", frame_bgr)
            self.save_frame = False

        dpg.set_value(
            "_texture", buffer_image
        )  # buffer must be contiguous, else seg fault!
        
        dpg.set_value("_log_mouse_value", f"({[f'{v:.4f}' for v in mous_hover_value]})")

        # Add _log_view_camera
        dpg.set_value("_log_view_camera", f"View {self.current_cam_index}")
        if 1./(t1-t0) < 500:
            dpg.set_value("_log_infer_time", f"{1./(t1-t0)} ")

    @torch.no_grad()
    def play_video(self):
        cap = cv2.VideoCapture(self.novel_view_background_dir)
        to_tensor = transforms.ToTensor()
        recorded_frames = []            # stored new frames
        output_fps = 15     

        if self.save_custom_video:
            view_fps = 120
        else:
            view_fps = 30
        view_fps = 200

        output_fps = cap.get(cv2.CAP_PROP_FPS)
        
        while self.play_custom_video:
            ret, frame = cap.read()
            if not ret or frame is None:
                self.play_custom_video = False
                break                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = to_tensor(frame).cuda()

            cam = self.free_cams[self.current_cam_index]
            cam.time = self.time
            abc = self.abc.abc
            mask = cam.sceneoccluded_mask.squeeze(0).cuda()

            buffer_image = render(
                cam,
                self.gaussians,
                abc,
                tensor,
                view_args={
                    "vis_mode": 'render',
                    "stage": 'fine',
                    "finecoarse_flag": False
                },
                mip_level=self.opt.mip_level,
                blending_mask=mask
            )["render"]
        
            
            buffer_image = torch.nn.functional.interpolate(
                buffer_image.unsqueeze(0),
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)

            arr = (
                buffer_image.permute(1, 2, 0)
                .clamp(0, 1)
                .detach()
                .cpu()
                .numpy()
                .copy()
            )

            dpg.set_value("_texture", arr)
            # time.sleep(1./view_fps)
            dpg.render_dearpygui_frame()
            
            if self.save_custom_video:
                # convert float rgb → u+nt8 bgr for OpenCV video writing
                frame_uint8 = (arr * 255).astype("uint8")
                frame_bgr = cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR)
                recorded_frames.append(frame_bgr)
        
        if self.save_custom_video:
            if len(recorded_frames) == 0:
                print("No frames recorded, skipping save.")
                return

            output_path = os.path.join(os.path.join(self.save_videos, f"custom_video.mp4"))
            h, w, _ = recorded_frames[0].shape

            print(f"Saving custom video → {output_path}")
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                (w, h)
            )

            for f in recorded_frames:
                writer.write(f)

            writer.release()
            print("Video saved.")
            self.save_custom_video = False
        cap.release()
    
    @torch.no_grad()
    def initialize_abc(self):
        import torch
        from scene.dataset import ABC
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import json
        # Initialize ABC
        # TL, TR, BL
        # Scene 1
        init = torch.tensor([[-0.152, 0.73, 0.174],
                             [0.461, 0.50, 0.18],
                             [-0.126, 0.79, -0.18]])
        
        # init = torch.tensor([[-0.8742,1.819,-0.1954],
        #                      [1.377, 1.456,0.310],
        #                      [-0.976,1.739,0.368]])
        
        # Check to see if data is stored on the ibl pose:
  
        with open(os.path.join(self.args.source_path,'transforms.json')) as dict_json:
            dic = json.load(dict_json)
            
        if "ibl_abc" not in dic.keys():
            # Project image infrom of first camera
            init = torch.tensor([[-0.976,1.739,0.368],
                                [1.377, 1.456,0.310],
                                 [-0.8742,1.819,-0.1954]
                                ])
        else:
            init = torch.tensor(dic["ibl_abc"])

        print(f"Texture ABC : {init}")
        self.abc = ABC(init.cuda())

    def save_scene(self):
        print("\n[ITER {}] Saving Gaussians".format(self.iteration))
        self.scene.save(self.iteration, self.stage)
        print("\n[ITER {}] Saving Checkpoint".format(self.iteration))
        torch.save((self.gaussians.capture(), self.iteration), self.scene.model_path + "/chkpnt" + f"_" + str(self.iteration) + ".pth")

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")
            
        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=400,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("N/A", tag="_log_infer_time")
            with dpg.group(horizontal=True):
                dpg.add_text("Stage: ")
                dpg.add_text("N/A", tag="_log_stage")
            # ----------------
            #  Loss Functions
            # ----------------
            with dpg.group():
                if self.view_test is False:
                    dpg.add_text("Training info:")
                    dpg.add_text("N/A", tag="_log_iter")
                    dpg.add_text("N/A", tag="_log_relit")
                    dpg.add_text("N/A", tag="_log_canon")
                    dpg.add_text("N/A", tag="_log_deform")
                    dpg.add_text("N/A", tag="_log_plane")
                    dpg.add_text("N/A", tag="_log_fine2")

                    dpg.add_text("N/A", tag="_log_points")
                    with dpg.collapsing_header(label="Testing info:", default_open=True):
                        with dpg.group(horizontal=True):
                            dpg.add_text("Progress : ")
                            dpg.add_text("N/A", tag="_log_test_progress")

                        dpg.add_text("Lighting-only : ")
                        dpg.add_text("N/A", tag="_log_l_1")
                        dpg.add_text("N/A", tag="_log_l_2")
                        dpg.add_text("N/A", tag="_log_l_3")
                        dpg.add_text("N/A", tag="_log_l_4")
                        dpg.add_text("N/A", tag="_log_l_5")
                        dpg.add_text("View-only : ")
                        dpg.add_text("N/A", tag="_log_v_1")
                        dpg.add_text("N/A", tag="_log_v_2")
                        dpg.add_text("N/A", tag="_log_v_3")
                        dpg.add_text("View & Lighting : ")
                        dpg.add_text("N/A", tag="_log_lv_1")
                        dpg.add_text("N/A", tag="_log_lv_2")
                        dpg.add_text("N/A", tag="_log_lv_3")
                        dpg.add_text("N/A", tag="_log_lv_4")
                        dpg.add_text("N/A", tag="_log_lv_5")
                    
                else:
                    dpg.add_text("Mode : viewing")

            # ----------------
            #  Control Functions
            # ----------------
            with dpg.collapsing_header(label="Viewer Config", default_open=True):
                def callback_viewer_on(sender):
                    self.switch_off_viewer = False
                    
                def callback_viewer_off(sender):
                    self.switch_off_viewer = True
                    
                def callback_on_finecoarse(sender):
                    self.finecoarse_flag = False
                def callback_off_finecoarse(sender):
                    self.finecoarse_flag = True
                dpg.add_text(" : Main Settings : ")
                with dpg.group(horizontal=True):
                    dpg.add_text("Viewer : ")
                    dpg.add_button(label="On", callback=callback_viewer_on)  
                    dpg.add_button(label="Off", callback=callback_viewer_off)  
                    dpg.add_text(" | Relighting : ")
                    dpg.add_button(label="On", callback=callback_on_finecoarse)
                    dpg.add_button(label="Off", callback=callback_off_finecoarse)
                   
                     
                def callback_toggle_reset_cam(sender):
                    self.current_cam_index = 0
                    
                def callback_toggle_next_cam(sender):
                    self.current_cam_index = (self.current_cam_index + 1) % len(self.free_cams)
                def callback_toggle_before_cam(sender):
                    diff = (self.current_cam_index - 1)
                    if diff < 0: 
                        self.current_cam_index = len(self.free_cams) -1
                    else:
                        self.current_cam_index = diff % len(self.free_cams)
                
                with dpg.group(horizontal=True):
                    dpg.add_button(label="<<", callback=callback_toggle_before_cam)
                    dpg.add_text("N/A", tag="_log_view_camera")
                    dpg.add_button(label=">>", callback=callback_toggle_next_cam)

                def callback_toggle_reset_cam(sender):
                    for i in range(len(self.free_cams)):
                        self.free_cams[i] = copy.deepcopy(self.original_cams[i])
                    self.current_cam_index = 0
                
                def callback_toggle_save_frame(sender):
                        self.save_frame = True
                dpg.add_text(": Frame Settings : ")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="save", callback=callback_toggle_save_frame)

                    dpg.add_button(label="reset", callback=callback_toggle_reset_cam)
                    dpg.add_button(label="reset (FOV)", callback=callback_toggle_reset_cam)

                def callback_toggle_sceneocc_mask(sender):
                    self.show_mask = 'occ'
                def callback_toggle_no_mask(sender):
                    self.show_mask = 'none'

                with dpg.group(horizontal=True):
                    dpg.add_text("Masks : ")
                    dpg.add_button(label="On", callback=callback_toggle_sceneocc_mask)
                    dpg.add_button(label="Off", callback=callback_toggle_no_mask)
                    
                
                def callback_toggle_show_rgb(sender):
                    self.vis_mode = 'render'
                def callback_toggle_show_alpha(sender):
                    self.vis_mode = 'alpha'
                def callback_toggle_show_depth(sender):
                    self.vis_mode = 'D'
                def callback_toggle_show_edepth(sender):
                    self.vis_mode = 'ED'
                def callback_toggle_show_2dgsdepth(sender):
                    self.vis_mode = '2D' 
                def callback_toggle_show_XYZ(sender):
                    self.vis_mode = 'xyz'
                def callback_toggle_show_invariance(sender):
                    self.vis_mode = 'invariance'
                def callback_toggle_show_uv(sender):
                    self.vis_mode = 'uv'
                def callback_toggle_show_sigma(sender):
                    self.vis_mode = 'sigma'
                def callback_toggle_show_deform(sender):
                    self.vis_mode = 'deform'

                dpg.add_text(" : Appearance Buffers : ")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="RGB", callback=callback_toggle_show_rgb)
                    dpg.add_button(label="A", callback=callback_toggle_show_alpha)
                    dpg.add_button(label="inv", callback=callback_toggle_show_invariance)
                    dpg.add_button(label="muv", callback=callback_toggle_show_uv)
                    dpg.add_button(label="sca", callback=callback_toggle_show_sigma)
                    dpg.add_button(label="def", callback=callback_toggle_show_deform)
                
                dpg.add_text(" : Geometry Buffers : ")
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Zc", callback=callback_toggle_show_depth)
                    dpg.add_button(label="E[Zc]", callback=callback_toggle_show_edepth)
                    dpg.add_button(label="Zc", callback=callback_toggle_show_2dgsdepth)
                    dpg.add_button(label="XYZ", callback=callback_toggle_show_XYZ)

                
                def callback_speed_control(sender):
                    self.time = int(dpg.get_value(sender)*100)
                dpg.add_text(" : Select IBL (image) : ")
                dpg.add_slider_float(
                    label="IBL",
                    default_value=0.,
                    max_value=1.,
                    min_value=0.,
                    callback=callback_speed_control,
                )
            
            if self.view_test:
                with dpg.collapsing_header(label="Rendering Processing", default_open=True):
                    
                    def callback_iblpose_on(sender):
                        self.drag_func = "ibl-pose"
                    def callback_iblpose_off(sender):
                        self.drag_func = "viewing"

                    dpg.add_text(" : Configure IBL : ")
                    with dpg.group(horizontal=True):
                        dpg.add_text("Pose IBL : ")
                        dpg.add_button(label="On" , callback=callback_iblpose_on)
                        dpg.add_button(label="Off" , callback=callback_iblpose_off)

                    def callback_iblpose_save(sender):
                        # Open the transforms file
                        with open(os.path.join(self.args.source_path,'transforms.json')) as dict_json:
                            dic = json.load(dict_json)

                        dic["ibl_abc"] = self.abc.abc.cpu().tolist()
                        with open(os.path.join(self.args.source_path,'transforms.json'), "w") as dict_json:
                            dic = json.dump(dic, dict_json)
                        
                    with dpg.group(horizontal=True):
                        dpg.add_text("Save pose : ")
                        dpg.add_button(label="save" , callback=callback_iblpose_save)
                    
                    def on_text_change(sender, app_data, user_data):
                        self.novel_view_background_dir = app_data
                        
                    def callback_toggle_render_novel_view(sender):
                        self.switch_off_viewer = True
                        self.play_custom_video = True

                    def callback_toggle_render_novel_view_save(sender):
                        self.switch_off_viewer = True
                        self.play_custom_video = True
                        self.save_custom_video = True
                    
                    dpg.add_text(" : Inset video IBL : ") 
                    with dpg.group(horizontal=True):
                        dpg.add_text("Filepath : ")
                        dpg.add_input_text(callback=on_text_change)
                    
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Play" , callback=callback_toggle_render_novel_view)
                        dpg.add_button(label="Play & Save" , callback=callback_toggle_render_novel_view_save)
        
        def zoom_callback_fov(sender, app_data):
            delta = app_data  # scroll: +1 = up (zoom in), -1 = down (zoom out)
            cam = self.free_cams[self.current_cam_index]

            zoom_scale = 0.8  # Smaller = faster zoom

            # Scale FoV within limits
            cam.fx *= zoom_scale if delta > 0 else 1 / zoom_scale
            cam.fy *= zoom_scale if delta > 0 else 1 / zoom_scale
            cam.update_K()

        def drag_callback(sender, app_data):
            # if we haven't initialized the drag image buffer
            if self.drag_im_buffer is not None:
                mouse_hover_value = self.drag_im_buffer[:3, self.mous_loc[1], self.mous_loc[0]].sum()
            else:
                mouse_hover_value = 0.
            view_drag_thresh = 0.5
            
            if mouse_hover_value < view_drag_thresh:
                button, rel_x, rel_y = app_data
            
                if button != 0:  # only left drag
                    return

                # simply check inside primary window dimensions
                if dpg.get_active_window() != dpg.get_alias_id("_primary_window"):
                    return
                
                cam = self.free_cams[self.current_cam_index]

                if not hasattr(cam, "yaw"): cam.yaw = 0.0
                if not hasattr(cam, "pitch"): cam.pitch = 0.0
        
                # Sensitivity
                yaw_speed = 0.0001
                pitch_speed = 0.0001

                cam.yaw = rel_x * yaw_speed
                cam.pitch = -rel_y * pitch_speed
                cam.pitch = np.clip(cam.pitch, -np.pi/2 + 0.01, np.pi/2 - 0.01)  # avoid flip

                # --- Rebuild rotation matrix from angles ---
                cy, sy = np.cos(cam.yaw), np.sin(cam.yaw)
                cp, sp = np.cos(cam.pitch), np.sin(cam.pitch)

                # Yaw (around world Y), Pitch (around local X)
                Ry = np.array([
                    [cy, 0, sy],
                    [0, 1, 0],
                    [-sy, 0, cy]
                ], dtype=np.float32)

                Rx = np.array([
                    [1, 0, 0],
                    [0, cp, -sp],
                    [0, sp, cp]
                ], dtype=np.float32)

                cam.R = cam.R @ Ry @ Rx 
            
            if self.drag_func == "ibl-pose" and mouse_hover_value > view_drag_thresh:
                # Need to do mouse-ray intersection to see if mouse lands on a abc
                # Render top-layer for ibl-pose
                # cache rgba of the Gsplat render, use the mouse-intersecting color to select a point to change
                # as mouse moves, move the point w.r.t fixed distance from the camera
                
                # Select a vertex to move
                mouse_hover_value = self.drag_im_buffer[:3, self.mous_loc[1], self.mous_loc[0]]
                if mouse_hover_value[0] > view_drag_thresh: # Top left
                    abc_index = 0
                elif mouse_hover_value[1]> view_drag_thresh: # Top Right
                    abc_index = 1
                elif mouse_hover_value[2]> view_drag_thresh: # Bottom Left
                    abc_index = 2
                
                xyz = self.abc.abc[abc_index].unsqueeze(-1)

                # Project into 2-D to find the center w.r.t image space
                cam = self.free_cams[self.current_cam_index]
                intr = cam.intrinsics # 3x3
                w2c = cam.w2c # 4x4
                width = cam.image_width
                height = cam.image_height
                
                R = w2c[:3, :3]
                T = w2c[:3, 3:4]
                Xc = R @ xyz + T
                z = Xc[2]
                
                fx, fy = intr[0,0], intr[1,1]
                cx,cy = intr[0,2], intr[1,2]
                
                # Get 2-D center for selected abc
                u = fx*(Xc[0]/z) + cx
                v = fy*(Xc[1]/z) + cy 
                
                dx = self.mous_loc[0] - self.mous_loc_last[0]
                dy = self.mous_loc[1] - self.mous_loc_last[1]
                # dx, dy = dpg.get_mouse_drag_delta()
                u_ = u + dx
                v_ = v + dy
    
                x_ = (u_ - cx)/fx
                y_ = (v_ - cy)/fy
                
                Xc_ = torch.tensor([x_*z, y_*z, z]).unsqueeze(-1).cuda()
                Xw = R.T @ (Xc_-T)
                self.abc.abc[abc_index] = Xw.squeeze(-1)
                            
        def mouse_hover_callback(sender, app_data):
            # app_data: (x, y) coordinates of mouse position in global viewport
            x, y = app_data

            if dpg.is_item_hovered("_primary_window"):
                self.mous_loc_last = self.mous_loc
                self.mous_loc = [int(x),int(y)]
                dpg.set_value("_log_mouse_xy", f"({x:.1f}, {y:.1f})")

        with dpg.group(horizontal=True, parent="_control_window"):
            dpg.add_text(" : Mouse data : ")
        # Add text in the control window to display mouse coordinates
        with dpg.group(horizontal=True, parent="_control_window"):
            dpg.add_text("Position : ")
            dpg.add_text("N/A", tag="_log_mouse_xy")
        with dpg.group(horizontal=True, parent="_control_window"):
            dpg.add_text("Pixel Value : ")
            dpg.add_text("N/A", tag="_log_mouse_value")
            
        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=zoom_callback_fov)
            dpg.add_mouse_drag_handler(callback=drag_callback)
            dpg.add_mouse_move_handler(callback=mouse_hover_callback)
            
            
        dpg.create_viewport(
            title=f"{self.runname}",
            width=self.W + 400,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        
        
            
        dpg.setup_dearpygui()

        dpg.show_viewport()
        
from scipy.ndimage import distance_transform_edt
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

from scipy.ndimage import distance_transform_edt
@torch.no_grad()
def get_in_view_dyn_mask(camera, xyz, X, Y) -> torch.Tensor:
    device = xyz.device
    N = xyz.shape[0]

    # Convert to homogeneous coordinates
    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)

    # World → Camera (OpenCV convention: +Z forward)
    c2w = camera.pose
    w2c = get_viewmat(c2w[None])[0]
    xyz_cam = (xyz_h @ w2c.T)[:, :3]

    # Only keep points in front of the camera
    in_front = xyz_cam[:, 2] > 0

    # Camera → Pixel (using intrinsics)
    K = torch.from_numpy(camera.K).to(device=device, dtype=torch.float32)
    xy = xyz_cam @ K.T  # (N, 3)
    px = (xy[:, 0] / xy[:, 2]).long()
    py = (xy[:, 1] / xy[:, 2]).long()

    # Visibility check (inside image bounds)
    in_bounds = (
        (px >= 0) & (px < camera.image_width) &
        (py >= 0) & (py < camera.image_height)
    )
    visible_mask = in_front & in_bounds

    # Valid pixel indices
    valid_idx = visible_mask.nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        print("No visible points found.")
        return torch.zeros((camera.image_height, camera.image_width, 3), device=device)

    px_valid = px[valid_idx]
    py_valid = py[valid_idx]

    # Scene occlusion mask (optional)
    mask = (1. - camera.sceneoccluded_mask).to(device).squeeze(0)

    sampled_mask = mask[py_valid, px_valid] > 0.5

    # Projected XYZ image
    H, W = camera.image_height, camera.image_width
    xyz_img = torch.zeros((H, W, 3), device=device)

    px_final = px_valid[sampled_mask]
    py_final = py_valid[sampled_mask]
    xyz_vals = xyz[valid_idx][sampled_mask]
    xyz_img[py_final, px_final] = xyz_vals

    # Visualization (optional)
    show = False
    if show:
        import matplotlib.pyplot as plt
        img_np = xyz_img.detach().cpu().numpy()

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(img_np)
        ax[0].set_title("Projected XYZ (OpenCV)")
        ax[0].axis("off")

        ax[1].imshow(img_np)
        ax[1].scatter(X, Y, s=3, c="red")
        ax[1].set_title("With XY indexing")
        ax[1].axis("off")

        plt.show()
        exit()
        return None
    
    # --- Nearest-neighbor fill for empty pixels ---
    xyz_np = xyz_img.cpu().numpy()   # [H, W, 3]
    valid_mask = (xyz_np.sum(axis=-1) != 0)

    # distance_transform_edt returns for each empty pixel the index of the nearest valid pixel
    dist, indices = distance_transform_edt(~valid_mask,
                                           return_indices=True)
    filled = xyz_np[indices[0], indices[1]]  # nearest xyz per pixel

    point = filled[Y, X, :]
    
    show = False
    if show:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter all visible point cloud
        ax.scatter(
            xyz_vals[:, 0].cpu(),
            xyz_vals[:, 1].cpu(),
            xyz_vals[:, 2].cpu(),
            s=1, c="blue", alpha=0.5, label="Point cloud"
        )

        # Scatter your selected points
        ax.scatter(
            point[ 0],
            point[ 1],
            point[ 2],
            s=60, c="red", marker="o", label="Filtered XYZ"
        )

        ax.set_title("3D Point Cloud with Filtered Points")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        plt.show()
        exit()

    return torch.from_numpy(point).float().to(device)


def remove_screen_points(camera, xyz):
    device = xyz.device
    N = xyz.shape[0]

    # Convert to homogeneous coordinates
    xyz_h = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=-1)  # (N, 4)

    # Apply full projection (world → clip space)
    proj_xyz = xyz_h @ camera.full_proj_transform.to(device)  # (N, 4)

    # Homogeneous divide to get NDC coordinates
    ndc = proj_xyz[:, :3] / proj_xyz[:, 3:4]  # (N, 3)

    in_front = proj_xyz[:, 2] > 0
    in_ndc_bounds = (
        (ndc[:, 0].abs() <= 1) &
        (ndc[:, 1].abs() <= 1) &
        (ndc[:, 2].abs() <= 1)
    )
    visible_mask = in_front & in_ndc_bounds

    # Pixel coordinates for all points (will clamp to bounds)
    px = (((ndc[:, 0] + 1) / 2) * camera.image_width).long().clamp(0, camera.image_width - 1)
    py = (((ndc[:, 1] + 1) / 2) * camera.image_height).long().clamp(0, camera.image_height - 1)

    # Scene mask (1 = free, 0 = masked/occluded)
    mask_img = (camera.sceneoccluded_mask).to(device).squeeze(0)

    # Start with all points marked False (not removed)
    remove_mask = torch.zeros(N, dtype=torch.bool, device=device)

    # Only check points that are visible
    sampled_mask = mask_img[py[visible_mask], px[visible_mask]].bool()

    # Mark visible points inside the mask for removal
    remove_mask[visible_mask] = sampled_mask

    return remove_mask
    

