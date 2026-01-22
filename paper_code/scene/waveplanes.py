from typing import Optional

from submodules.pytorch_wavelets_.dwt.transform2d import DWTInverse

from utils.kplane_utils import grid_sample_wrapper, normalize_aabb, GridSet

import torch
import torch.nn as nn
import torch.nn.functional as F

# OFFSETS = torch.tensor([
#     [-1.0, 0.0],
#     [-0.5, 0.0],
#     [0.5, 0.0],
#     [1.0, 0.0],
#     [0.0, -1.0],
#     [0.0, -0.5],
#     [0.0, 0.5],
#     [0.0, 1.0],
#     [0.5, 0.5],
#     [0.5, -0.5],
#     [-0.5, 0.5],
#     [-0.5, -0.5]
# ]).cuda().unsqueeze(0)

def build_cov_matrix_torch(cov6):
    N = cov6.shape[0]
    cov = torch.zeros((N, 3, 3), device=cov6.device, dtype=cov6.dtype)
    cov[:, 0, 0] = cov6[:, 0]  # σ_xx
    cov[:, 0, 1] = cov[:, 1, 0] = cov6[:, 1]  # σ_xy
    cov[:, 0, 2] = cov[:, 2, 0] = cov6[:, 2]  # σ_xz
    cov[:, 1, 1] = cov6[:, 3]  # σ_yy
    cov[:, 1, 2] = cov[:, 2, 1] = cov6[:, 4]  # σ_yz
    cov[:, 2, 2] = cov6[:, 5]  # σ_zz
    return cov

def sample_from_cov(points, cov6, M):
    N = points.shape[0]
    cov = build_cov_matrix_torch(cov6)  # (N, 3, 3)
    
    # Cholesky decomposition: cov = L @ L.T
    L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(3, device=points.device))  # (N, 3, 3)
    
    # Sample standard normal noise: (N, M, 3)
    eps = torch.randn(N, M, 3, device=points.device)
    
    # Transform noise by covariance and add mean
    # L: (N, 3, 3), eps: (N, M, 3) → (N, M, 3)
    samples = torch.einsum('nij,nmj->nmi', L, eps) + points[:, None, :]
    
    return samples 

def interpolate_features_MUL(data, kplanes):
    """Generate features for each point
    """
    # time m feature
    space = 1.
    
    # q,r are the coordinate combinations needed to retrieve pts
    coords = [[0,1], [0,2], [1,2]]
    for i in range(len(coords)):
        q,r = coords[i]
        feature = kplanes[i](data[..., (q, r)])
        space = space * feature

    return space
   

import matplotlib.pyplot as plt

def visualize_grid_and_coords(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True):
    """
    Visualizes the mean grid (averaged over batch) and overlays the sampling coordinates.
    
    Args:
        grid (torch.Tensor): Input tensor of shape [1, B, H, W] or [B, H, W].
        coords (torch.Tensor): Normalized coordinates in [-1, 1] of shape [N, 2] or [1, N, 2].
        align_corners (bool): Whether the coords use align_corners=True (affects projection).
    """
    # Remove singleton channel dimension if present (e.g. [1, B, H, W] -> [B, H, W])
    if grid.dim() == 4 and grid.shape[0] == 1:
        grid = grid.squeeze(0)
    
    if grid.dim() != 3:
        raise ValueError("Expected grid shape [B, H, W]")

    # Mean across batch axis
    grid_mean = grid.mean(dim=0)  # [H, W]

    H, W = grid_mean.shape
    # Handle coordinate dimensions
    if coords.dim() == 3:
        coords = coords.squeeze(0)  # [N, 2]

    if coords.shape[-1] != 2:
        raise ValueError("Coordinates must be 2D")
    
    # Convert normalized coordinates [-1, 1] to image coordinates
    def denorm_coords(coords, H, W, align_corners):
        if align_corners:
            x = ((coords[:, 0] + 1) / 2) * (W - 1)
            y = ((coords[:, 1] + 1) / 2) * (H - 1)
        else:
            x = ((coords[:, 0] + 1) * W - 1) / 2
            y = ((coords[:, 1] + 1) * H - 1) / 2
        return x, y

    x, y = denorm_coords(coords, H, W, align_corners)

    # exit()
    # Plotting
    plt.figure(figsize=(6, 6))
    plt.imshow(grid_mean.cpu().numpy(), cmap='gray', origin='upper')
    plt.scatter(x.cpu().numpy(), y.cpu().numpy(), color='red', s=20)
    plt.title("Grid Mean with Sampled Coords")
    plt.axis('off')
    plt.show()

class WavePlaneField(nn.Module):
    def __init__(
            self,
            bounds,
            planeconfig, 
            rotate=False
    ):
        super().__init__()
        aabb = torch.tensor([[bounds, bounds, bounds],
                             [-bounds, -bounds, -bounds]])
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.concat_features = True
        self.grid_config = planeconfig
        self.feat_dim = self.grid_config["output_coordinate_dim"]

        # 1. Init planes
        self.grids = nn.ModuleList()

        # Define the DWT functon
        self.cacheplanes = True
        self.is_waveplanes = True
        
        for i in range(3):
            res = [self.grid_config['resolution'][0],
                self.grid_config['resolution'][0]]
            what = "space"
            
            gridset = GridSet(
                what=what,
                resolution=res,
                J=self.grid_config['wavelevel'],
                config={
                    'feature_size': self.grid_config["output_coordinate_dim"],
                    'a': 0.1,
                    'b': 0.5,
                    'wave': 'coif4',
                    'wave_mode': 'periodization',
                },
                cachesig=self.cacheplanes
            )

            self.grids.append(gridset)

    @property
    def get_aabb(self):
        return self.aabb[0], self.aabb[1]

    def set_aabb(self, xyz_max, xyz_min):
        try:
            aabb = torch.tensor([
                xyz_max,
                xyz_min
            ], dtype=torch.float32)
        except:
            aabb = torch.stack([xyz_max, xyz_min], dim=0)  # Shape: (2, 3)
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        print("Voxel Plane: set aabb=", self.aabb)

    def waveplanes_list(self):
        planes = []
        for grid in self.grids:
            planes.append(grid.grids)
        return planes
    
    def grids_(self, regularise_wavelet_coeff: bool = False, time_only: bool = False, notflat: bool = False):
        """Return the grids as a list of parameters for regularisation
        """
        ms_planes = []
        for i in range(len(self.grids)):
            # if i < 6:
            gridset = self.grids[i]

            ms_feature_planes = gridset.signal

            # Initialise empty ms_planes
            if ms_planes == []:
                ms_planes = [[] for j in range(len(ms_feature_planes))]

            for j, feature_plane in enumerate(ms_feature_planes):
                ms_planes[j].append(feature_plane)

        return ms_planes

    def forward(self, pts):
                
        pts = normalize_aabb(pts, self.aabb)
        pts = pts.reshape(-1, 3)

        return interpolate_features_MUL(
            pts, self.grids)

