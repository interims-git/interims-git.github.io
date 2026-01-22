import torch
import numpy as np
from torch import nn
import os

from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement

from utils.general_utils import strip_symmetric, build_scaling_rotation

from gsplat import DefaultStrategy, MCMCStrategy
from plyfile import PlyData, PlyElement

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid

        self.sigmoid_activation = torch.sigmoid
        
        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, args):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        
        self.gsplat_optimizers = None
        
        self.spatial_lr_scale = 0
        self.spatial_lr_scale_background = 0
        self.target_neighbours = None
        
        self.use_default_strategy = False
        
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self.splats,
            self.spatial_lr_scale,
            self.spatial_lr_scale_background
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self.splats,
        self.spatial_lr_scale,self.spatial_lr_scale_background) = model_args
        
        self.training_setup(training_args)

    @torch.no_grad()
    def compute_3D_filter(self, cameras):
        #TODO consider focal length and image width
        xyz = self.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)
        
        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        for camera in cameras:

            # transform points to camera space
            R = torch.tensor(camera.R, device=xyz.device, dtype=torch.float32)
            T = torch.tensor(camera.T, device=xyz.device, dtype=torch.float32)
             # R is stored transposed due to 'glm' in CUDA code so we don't neet transopse here
            xyz_cam = xyz @ R + T[None, :]
                        
            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.1
            
            
            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)
            
            x = x / z * camera.fx + camera.image_width / 2.0
            y = y / z * camera.fy + camera.image_height / 2.0
            
            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.image_width), torch.logical_and(y >= 0, y < camera.image_height))
            
            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.image_width, x <= camera.image_width * 1.15), torch.logical_and(y >= -0.15 * camera.image_height, y <= 1.15 * camera.image_height))
            
        
            valid = torch.logical_and(valid_depth, in_screen)
            
            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            distance[valid] = torch.min(distance[valid], z[valid])
            valid_points = torch.logical_or(valid_points, valid)
            if focal_length < camera.fx:
                focal_length = camera.fx
        
        distance[~valid_points] = distance[valid_points].max()
        #TODO box to gaussian transform
        filter_3D = distance / focal_length * (0.2 ** 0.5)
        # self.filter_3D = filter_3D[..., None]

    @property
    def get_features(self):
        features_dc = self.splats['sh0']
        features_rest = self.splats['shN']
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_lambda(self):
        features_dc = self.splats['lambda_sh0']
        features_rest = self.splats['lambda_shN']
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_ab(self):
        features_dc = self.splats['ab_sh0']
        features_rest = self.splats['ab_shN']
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_texscale(self):
        return torch.sigmoid(self.splats["tex_scale"])
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self.splats["scales"])
    

    @property
    def get_rotation(self):
        return self.rotation_activation(self.splats["quats"])

    @property
    def get_xyz(self):
        return self.splats["means"]

    @property
    def get_opacity(self):
        return torch.sigmoid(self.splats["opacities"])

    
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)
    
    @property
    def get_covmat(self):
        w, x, y, z = self.get_rotation.unbind(-1)
        scale = self.get_scaling
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z

        R = torch.stack([
            torch.stack([1 - 2*(yy+zz), 2*(xy - wz),     2*(xz + wy)], dim=-1),
            torch.stack([2*(xy + wz),   1 - 2*(xx+zz),   2*(yz - wx)], dim=-1),
            torch.stack([2*(xz - wy),   2*(yz + wx),     1 - 2*(xx+yy)], dim=-1),
        ], dim=-2)
        
        e1 = torch.tensor([1,0,0], device=scale.device, dtype=scale.dtype).expand(scale.size(0), -1)  # (N,3)
        e2 = torch.tensor([0,1,0], device=scale.device, dtype=scale.dtype).expand(scale.size(0), -1)  # (N,3)

        # Scale local basis
        v1 = e1 * scale[:, [0]]  # (N,3)
        v2 = e2 * scale[:, [1]]  # (N,3)

        # Apply rotation: batch matmul (N,3,3) @ (N,3,1) -> (N,3,1)
        t_u = torch.bmm(R, v1.unsqueeze(-1)).squeeze(-1)  # (N,3)
        t_v = torch.bmm(R, v2.unsqueeze(-1)).squeeze(-1)  # (N,3)

        # Magnitudes
        m_u = torch.linalg.norm(t_u, dim=-1)
        m_v = torch.linalg.norm(t_v, dim=-1)

        # Directions (normalized)
        d_u = t_u / m_u.unsqueeze(-1)
        d_v = t_v / m_v.unsqueeze(-1)

        magnitudes = torch.stack([m_u, m_v], dim=-1)     # (N,2)
        directions = torch.stack([d_u, d_v], dim=1)      # (N,2,3)

        return magnitudes, directions

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def training_setup(self, training_args):        
        ##### Set-up GSplat optimizers #####        
        if self.use_default_strategy:
            self.strategy = DefaultStrategy(
                verbose=True,
                
                prune_opa=training_args.prune_opa,
                grow_grad2d=training_args.grow_grad2d, # GSs with image plane gradient above this value will be split/duplicated. Default is 0.0002.
                grow_scale3d=training_args.grow_scale3d,# GSs with 3d scale (normalized by scene_scale) below this value will be duplicated. Above will be split. Default is 0.01.
                grow_scale2d=training_args.grow_scale2d,# GSs with 2d scale (normalized by image resolution) above this value will be split. Default is 0.05.
                prune_scale3d=training_args.prune_scale3d,# GSs with 3d scale (normalized by scene_scale) above this value will be pruned. Default is 0.1.
                
                # refine_scale2d_stop_iter=4000, # splatfacto behavior
                refine_start_iter=training_args.densify_from_iter,
                refine_stop_iter=training_args.densify_until_iter,
                reset_every=training_args.opacity_reset_interval,
                refine_every=training_args.densification_interval,
                absgrad=False,
                revised_opacity=False,
                key_for_gradient="means2d",
                
            )
            self.strategy.check_sanity(self.splats, self.gsplat_optimizers)
            self.strategy_state = self.strategy.initialize_state(
                    scene_scale=self.spatial_lr_scale
            )
        else:
            self.strategy = MCMCStrategy(
                cap_max=2_000_000,            # optional ceiling on splats
                noise_lr=5e5,                 # strength of the random walk
                refine_start_iter=500,
                refine_stop_iter=training_args.densify_until_iter,
                refine_every=training_args.densification_interval,
                min_opacity=0.01,             # prune floor; match the rest of your pipeline
                verbose=True
            )
            self.strategy.check_sanity(self.splats, self.gsplat_optimizers)
            self.strategy_state = self.strategy.initialize_state()
        

    def pre_train_step(self, iteration, max_iterations, stage):
        """Run pre-training step functions"""
        if iteration % 100 == 0:
            self.oneupSHdegree()
            
        return None    

    def pre_backward(self, iteration, info):
        self.strategy.step_pre_backward(
            params=self.splats,
            optimizers=self.gsplat_optimizers,
            state=self.strategy_state,
            step=iteration,
            info=info,
            
        )
        
    def post_backward(self, iteration, info, stage):
        if self.use_default_strategy:
            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.gsplat_optimizers,
                state=self.strategy_state,
                step=iteration,
                info=info,
                packed=True,
            )
        else:
            means_lr = self.gsplat_optimizers["means"].param_groups[0]["lr"]
            self.strategy.step_post_backward(
                params=self.splats,
                optimizers=self.gsplat_optimizers,
                state=self.strategy_state,
                step=iteration,
                info=info,
                lr=means_lr,
            )

        for optimizer in self.gsplat_optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self.splats["sh0"].shape[1]*self.splats["sh0"].shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self.splats["shN"].shape[1]*self.splats["shN"].shape[2]):
            l.append('f_rest_{}'.format(i))
        for i in range(self.splats["opacities"].shape[1]):
            l.append('opacity_{}'.format(i))
        for i in range(self.splats["scales"].shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self.splats["quats"].shape[1]):
            l.append('rot_{}'.format(i))
            
        for i in range(self.splats["lambda_sh0"].shape[1]*self.splats["lambda_sh0"].shape[2]):
            l.append('lambda_dc_{}'.format(i))
        for i in range(self.splats["lambda_shN"].shape[1]*self.splats["lambda_shN"].shape[2]):
            l.append('lambda_rest_{}'.format(i))
            
        for i in range(self.splats["ab_sh0"].shape[1]*self.splats["ab_sh0"].shape[2]):
            l.append('ab_dc_{}'.format(i))
        for i in range(self.splats["ab_shN"].shape[1]*self.splats["ab_shN"].shape[2]):
            l.append('ab_rest_{}'.format(i))
            
        for i in range(self.splats["tex_scale"].shape[1]):
            l.append('texscale_{}'.format(i))

        return l
    
    def load_model(self, path):
        print("No hexplane model for this branch {}".format(path))
        
    def save_deformation(self, path):
        print("No hexplane model for this branch {}".format(path))
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.splats["means"].detach().cpu().numpy()
        opacities = self.splats["opacities"].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        scale = self.splats["scales"].detach().cpu().numpy()
        rotation = self.splats["quats"].detach().cpu().numpy()
        
        f_dc = self.splats["sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self.splats["shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        lambda_dc = self.splats["lambda_sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        lambda_rest = self.splats["lambda_shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        ab_dc = self.splats["ab_sh0"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        ab_rest = self.splats["ab_shN"].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        texscale = self.splats["tex_scale"].detach().cpu().numpy()

        
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation,
                                     lambda_dc, lambda_rest, ab_dc, ab_rest, texscale), axis=1)
        # attributes = np.concatenate((xyz, normals, colors, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path, training_args, cams=None, num_cams=19):
        
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        

        means = torch.tensor(xyz, dtype=torch.float, device="cuda")

        opac_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity")]
        opacities = np.zeros((xyz.shape[0], len(opac_names)))
        for idx, attr_name in enumerate(opac_names):
            opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        col_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("color")]
        col_names = sorted(col_names, key = lambda x: int(x.split('_')[-1]))
        cols = np.zeros((xyz.shape[0], len(col_names)))
        for idx, attr_name in enumerate(col_names):
            cols[:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))
        
        try: # Try to load the relighting parameters, if not we need to construct them ourselves
            lambda_dc = np.zeros((xyz.shape[0], 1, 1))
            lambda_dc[:, 0, 0] = np.asarray(plydata.elements[0]["lambda_dc_0"])
            # lambda_dc[:, 1, 0] = np.asarray(plydata.elements[0]["lambda_dc_1"])
            # lambda_dc[:, 2, 0] = np.asarray(plydata.elements[0]["lambda_dc_2"])

            lambda_dc = torch.tensor(features_dc, dtype=torch.float)
            
            extra_lambda_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("lambda_rest_")]
            extra_lambda_names = sorted(extra_lambda_names, key = lambda x: int(x.split('_')[-1]))
            lambda_extra = np.zeros((xyz.shape[0], len(extra_lambda_names)))
            for idx, attr_name in enumerate(extra_lambda_names):
                lambda_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            lambda_extra = lambda_extra.reshape((lambda_extra.shape[0], 1, (self.max_sh_degree + 1) ** 2 - 1))
            lambda_extra = torch.tensor(lambda_extra, dtype=torch.float)

            
            ab_dc = np.zeros((xyz.shape[0], 2, 1))
            ab_dc[:, 0, 0] = np.asarray(plydata.elements[0]["ab_dc_0"])
            ab_dc[:, 1, 0] = np.asarray(plydata.elements[0]["ab_dc_1"])
            ab_dc = torch.tensor(ab_dc, dtype=torch.float)

            extra_ab_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("ab_rest_")]
            extra_ab_names = sorted(extra_ab_names, key = lambda x: int(x.split('_')[-1]))
            ab_extra = np.zeros((xyz.shape[0], len(extra_ab_names)))
            for idx, attr_name in enumerate(extra_ab_names):
                ab_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
            ab_extra = ab_extra.reshape((ab_extra.shape[0], 2, (self.max_sh_degree + 1) ** 2 - 1))
            ab_extra = torch.tensor(ab_extra, dtype=torch.float)
            
            texscale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("tex_scale_")]
            texscale = np.zeros((xyz.shape[0], len(texscale_names)))
            for idx, attr_name in enumerate(texscale_names):
                texscale[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
            texscale = torch.tensor(texscale, dtype=torch.float)
        except:
            print('Loading previous relighting parameters failed (this is an error if you are loading a checkpoint but not if you are initializing)')
            
            template_sh0 = torch.zeros_like(torch.tensor(features_dc, dtype=torch.float))
            template_shN = torch.zeros_like(torch.tensor(features_extra, dtype=torch.float))
            
            lambda_dc = template_sh0[:, :1, :] + 0.01 # bias towards object color
            lambda_extra = template_shN[:, :1, :]

            ab_dc = template_sh0[:, :2, :] + 0.5 # Centre of mipmap
            ab_extra = template_shN[:, :2, :]
            
            texscale = torch.zeros_like(torch.tensor(opacities, dtype=torch.float)) # bias towards the low res image
            texscale = texscale + 0.01*torch.rand_like(texscale)
            texscale = torch.logit(texscale)

            #### Filter out points not in the physical scene ####
            num_train_frames = int(len(cams) / (num_cams-1))
            target_cams = [cam for idx, cam in enumerate(cams) if (idx % num_train_frames) == 0]

            xyz_mask = means[:, 0].clone()*0.
            for cam in target_cams:
                inds = cam.screenspace_xyz_search(means.cpu())
                xyz_mask[inds] += 1
            xyz_mask = (xyz_mask < 1).cuda()
            
            means = means[xyz_mask]
            xyz_mask = xyz_mask.cpu().numpy()
            scales = scales[xyz_mask]
            rots = rots[xyz_mask]
            opacities = opacities[xyz_mask]
            features_dc = features_dc[xyz_mask]
            features_extra = features_extra[xyz_mask]
            
            lambda_dc = lambda_dc[xyz_mask]
            lambda_extra = lambda_extra[xyz_mask]
            ab_dc = ab_dc[xyz_mask]
            ab_extra = ab_extra[xyz_mask]
            texscale = texscale[xyz_mask]

        
        mean_foreground = means.mean(dim=0).unsqueeze(0)
        dist_foreground = torch.norm(means - mean_foreground, dim=1)
        self.spatial_lr_scale = torch.max(dist_foreground).detach().cpu().numpy() /2.

        self.active_sh_degree = 3

        self.params = {
            ("means", nn.Parameter(means.requires_grad_(True)), training_args.position_lr_init ),
            ("scales", nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)), training_args.scaling_lr),
            ("quats", nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)), training_args.rotation_lr),
            ("opacities",nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True)), training_args.opacity_lr),
            ("sh0", nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)), training_args.feature_lr),
            ("shN", nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True)), training_args.feature_lr/20.),

            ("lambda_sh0", nn.Parameter(lambda_dc.cuda().transpose(1, 2).contiguous().requires_grad_(True)), training_args.lambda_lr),
            ("lambda_shN", nn.Parameter(lambda_extra.cuda().transpose(1, 2).contiguous().requires_grad_(True)), training_args.lambda_lr/20.),
            
            ("ab_sh0", nn.Parameter(ab_dc.cuda().transpose(1, 2).contiguous().requires_grad_(True)), training_args.tex_mu_lr),
            ("ab_shN", nn.Parameter(ab_extra.cuda().transpose(1, 2).contiguous().requires_grad_(True)), training_args.tex_mu_lr/20.),
            ("tex_scale",nn.Parameter(texscale.cuda().requires_grad_(True)), training_args.tex_s_lr),

            
        }
        self.splats = torch.nn.ParameterDict({n: v for n, v, _ in self.params}).cuda()
        
        import math
        batch_size = training_args.batch_size
        self.gsplat_optimizers = {
            name: torch.optim.Adam(
                [{"params": self.splats[name], "lr": lr * math.sqrt(batch_size)}],
                eps=1e-15 / math.sqrt(batch_size),
                betas=(1 - batch_size * (1 - 0.9), 1 - batch_size * (1 - 0.999)),
            )
            for name, _, lr in self.params
        }

from scipy.spatial import KDTree
import torch

def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)

    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)
