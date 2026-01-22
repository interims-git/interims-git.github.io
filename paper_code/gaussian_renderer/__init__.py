import torch
import torch.nn.functional as F

from gsplat.rendering import rasterization

from utils.sh_utils import eval_sh

import time
def process_Gaussians(pc):
    means3D = pc.get_xyz
    colors = pc.get_features
    
    opacity = pc.get_opacity

    scales = pc.get_scaling #pc.get_scaling_with_3D_filter
    
    rotations = pc.rotation_activation(pc.splats["quats"])
    
    return means3D, rotations, opacity, colors, scales

def process_full_Gaussians(pc):
    # Use existing function for processing canon
    means3D, rotations, opacity, colors, scales = process_Gaussians(pc)
    
    invariance = pc.get_lambda
    texsample = pc.get_ab
    texscale = pc.get_texscale
        
    return means3D, rotations, opacity, colors, scales, texsample, texscale, invariance

def rendering_pass(means3D, rotation, scales, opacity, colors, invariance, cam, sh_deg=3, mode="RGB+D"):
    if mode in ['normals', '2D']:
        gmode = 'RGB+D'
    elif mode == "invariance":
        colors = invariance.unsqueeze(0)
        gmode = 'RGB'
        sh_deg=None
    else:
        gmode = mode
    
    try:
        intr = []
        w2c = []
        for c in cam:
            intr.append(c.intrinsics.unsqueeze(0))
            
            viewmat = c.w2c.unsqueeze(0)
            w2c.append(viewmat)
            
        intr = torch.cat(intr, dim=0)
        w2c = torch.cat(w2c, dim=0)
        width = c.image_width
        height = c.image_height
        if means3D.dim() == 3:
            intr = intr.unsqueeze(1)
            w2c = w2c.unsqueeze(1)
    except:
        intr = cam.intrinsics.unsqueeze(0)
        viewmat = cam.w2c.unsqueeze(0)
        w2c = viewmat
        width = cam.image_width
        height = cam.image_height
    # Typical RGB render of base color
    colors, alphas, meta = rasterization(
        means3D, rotation, scales, opacity.squeeze(-1), colors,
        w2c.cuda(), 
        intr.cuda(),
        width, 
        height,
        
        render_mode=gmode,
        
        # rasterize_mode='antialiased',
        # eps2d=0.3,
        
        packed=False,
        near_plane=0.01,
        far_plane=1e10,
        sh_degree=sh_deg, #pc.active_sh_degree,
    )
        
    return colors, alphas, meta
import torch.nn.functional as F

@torch.no_grad()
def make_cam_rays(camera):
    """Generate pixel-center camera rays in world space using the same intrinsics as gsplat."""
    H, W = camera.image_height, camera.image_width
    device = camera.device

    # Use the same intrinsics as gsplat (camera.K from getOptimalNewCameraMatrix)
    K = camera.intrinsics.to(device)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Pixel-center coordinates
    xs = torch.arange(W, device=device, dtype=torch.float32) + 0.5
    ys = torch.arange(H, device=device, dtype=torch.float32) + 0.5
    jj, ii = torch.meshgrid(ys, xs, indexing="ij")  # [H,W]

    x = (ii - cx) / fx
    y = (jj - cy) / fy
    z = torch.ones_like(x)
    dirs_cam = torch.stack([x, y, z], dim=-1)  # [H,W,3]

    # We need the openGL c2w to align points rendered via openGL coordspace (even if we store our cameras as opencv)
    c2w = camera.pose.to(device)
    
    dirs_world = dirs_cam @ c2w[:3, :3].T
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)
    origins_world = c2w[:3, 3].expand_as(dirs_world)

    return origins_world, dirs_world

def render_IBL_source(camera, abc, texture):
    H, W = camera.image_height, camera.image_width

    # 1. Camera rays
    origin, direction = make_cam_rays(camera)

    origin = origin.reshape(-1, 3)
    direction = direction.reshape(-1, 3)
    
    a,b,c = abc
    # Calculate t intersection for each ray
    u = b-a
    v = c-a
    
    normal = torch.cross(u, v, dim=-1)
    normal = F.normalize(normal, dim=-1).unsqueeze(-1)

    # Intersect rays with plane
    nom = (a.unsqueeze(0) - origin) @ normal
    denom = (direction @ normal)
    valid = denom.abs() > 1e-8
    t = torch.empty_like(denom)
    t[valid] = (nom[valid] / denom[valid])
    x = origin + t * direction

    # Project onto u,v basis
    # u_len2 = (u * u).sum()
    # v_len2 = (v * v).sum()
    # d = x - a
    # u_coord = (d @ u) / u_len2   # [0,1] left→right
    # v_coord = (d @ v) / v_len2   # [0,1] bottom→top
    
    d = x - a                                             # (N,3)

    # Construct Gram matrix and its inverse
    UV = torch.stack([u, v], dim=1)                       # (3,2)
    G = UV.T @ UV                                          # (2,2)
    G_inv = torch.inverse(G)

    # Compute coefficients (u_coord, v_coord)
    coeffs = (G_inv @ (UV.T @ d.T)).T                      # (N,2)

    u_coord = coeffs[:, 0]
    v_coord = coeffs[:, 1]
    
    # Only sample rays within patch
    mask = (u_coord >= 0) & (u_coord <= 1) & (v_coord >= 0) & (v_coord <= 1)
    mask = mask & (t.squeeze() > 0)
    # uv = torch.stack([u_coord, v_coord], dim=-1)  # (N,2)
    uv = torch.stack([u_coord, v_coord], dim=-1)
    uv = (2.0 * uv - 1.0).unsqueeze(0).unsqueeze(2)

    sampled = F.grid_sample(
        texture.unsqueeze(0), uv,
        align_corners=False, mode="bilinear"
    ).squeeze(0).squeeze(-1).reshape(3, H, W)
    
    mask = mask.reshape(H, W).float()
    sampled = sampled * mask.unsqueeze(0)

    return sampled

@torch.no_grad
def render_ibl_pose_points(cam, abc, mip_level):
    # Generate abc as gaussians to render
    means = abc
    rgb = abc.clone()*0.
    rgb[[0,1,2],[0,1,2]] += 1.
    opacity = means[:, 0].unsqueeze(-1)*0. + 1.
    scales = abc.clone()*0. + .05
    rotation = torch.cat([abc*0., abc[:,0].unsqueeze(-1)*0. + 1.], dim=-1)
    
    render, alpha, meta = rendering_pass(means, rotation, scales, opacity, rgb, 
                                        None, cam, sh_deg=None, mode="RGB")
    
    rgba = torch.cat([render, alpha], dim=-1).squeeze(0).permute(2,0,1)
    return rgba


@torch.no_grad
def render(viewpoint_camera, pc, abc, texture, view_args=None, mip_level=2, blending_mask=None):
    """
    Render the scene for viewing
    """
    extras = None

    
    if view_args["finecoarse_flag"]:

        if view_args['vis_mode'] not in ['invariance', 'deform', 'uv', 'sigma']:
            means3D, rotation, opacity, colors, scales = process_Gaussians(pc)
            invariance = None
        else:
            means3D, rotation, opacity, colors, scales, texsample, texscale, invariance = process_full_Gaussians(pc)
            
            shs_view = texsample.transpose(1, 2).view(-1, 2, 16)
            dir_pp = (means3D - viewpoint_camera.camera_center.cuda().repeat(texsample.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            texsample_ab = torch.clamp_min(sh2rgb + 0.5, 0.0)
            
            shs_view = invariance.transpose(1, 2).view(-1, invariance.shape[-1], 16)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            invariance = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        active_sh = pc.active_sh_degree
        # Set arguments depending on type of viewing
        if view_args['vis_mode'] in 'render':
            mode = "RGB"
        elif view_args['vis_mode'] == 'alpha':
            mode = "RGB"
        elif view_args['vis_mode'] == 'normals':
            mode = "normals"
        elif view_args['vis_mode'] == '2D':
            mode = "2D"
        elif view_args['vis_mode'] == 'D':
            mode = "D"
        elif view_args['vis_mode'] == 'ED':
            mode = "ED"
        elif view_args['vis_mode'] in ['invariance', 'uv', 'sigma']:
            mode = "invariance"
            if view_args['vis_mode']  == 'uv':
                invariance = texsample_ab
            elif view_args['vis_mode']  == 'sigma':
                invariance = texscale
        elif view_args['vis_mode'] == 'xyz':
            mean_max = means3D.max()
            mean_min = means3D.min()
            colors = (means3D - mean_min) / (mean_max - mean_min)
            colors = means3D.unsqueeze(0)
            
        elif view_args['vis_mode'] == 'deform':
            colors = sample_mipmap(texture, texsample_ab, texscale, num_levels=2).unsqueeze(0)
        
        # Change for rendering with rgb instead of shs
        if view_args['vis_mode'] in ["deform", "xyz"]:
            mode = "RGB"
            active_sh=None
            
        # Render
        render, alpha, _ = rendering_pass(
            means3D, rotation, scales, opacity, colors, invariance,
            viewpoint_camera, 
            active_sh,
            mode=mode
        )
        
        if view_args['vis_mode'] == 'normals' or view_args['vis_mode'] == '2D':
            view_args['vis_mode'] = 'render'
        
        # Process image
        if view_args['vis_mode'] == 'render':
            render = render.squeeze(0).permute(2,0,1)

        elif view_args['vis_mode'] == 'alpha':
            render = alpha
            render = (render - render.min())/ (render.max() - render.min())
            render = render.squeeze(0).permute(2,0,1).repeat(3,1,1)
            
        elif view_args['vis_mode'] == 'D':
            render = (render - render.min())/ (render.max() - render.min())
            render = render.squeeze(0).permute(2,0,1).repeat(3,1,1)
            
        elif view_args['vis_mode'] == 'ED':
            render = (render - render.min())/ (render.max() - render.min())
            render = render.squeeze(0).permute(2,0,1).repeat(3,1,1)
            
        elif view_args['vis_mode'] == 'invariance':
            render = render.squeeze(0).permute(2,0,1).repeat(3,1,1)
        elif view_args['vis_mode'] == 'uv':
            render = render.squeeze(0).permute(2,0,1)
            render = torch.cat([render, render[0].unsqueeze(0)*0.], dim=0)
        elif view_args['vis_mode'] == 'sigma':
            render = render.squeeze(0).permute(2,0,1).repeat(3,1,1)*5.

        elif view_args['vis_mode'] in 'deform':
            render = render.squeeze(0).permute(2,0,1)
        
        elif view_args['vis_mode'] == 'xyz':
            render = render.squeeze(0).permute(2,0,1)
    else:

        render, _, alpha, _ = render_extended([viewpoint_camera], pc, [texture], mip_level=mip_level, exposure=view_args['exposure'])
        render = render.squeeze(0)
        alpha = alpha.squeeze(-1).squeeze(0)
        if abc is not None:
            # t1 = time.time()
            ibl = render_IBL_source(viewpoint_camera, abc, texture)
            # if blending_mask is not None:
            #     alpha = blending_mask.unsqueeze(0)

            
            render =  render * (alpha) + (1. - alpha) * ibl
            # t2 = time.time()
            # print( 1./(t2-t1))
        
    return {
        "render": render,
        "extras":extras # A dict containing mor point info
        }

def render_extended(viewpoint_camera, pc, textures, return_canon=False, mip_level=2, cli_use_canon=True, exposure=None):
    """Fine/Deformation function
    Notes:
        Trains/Renders the deformed gaussians
    """
    # t1 = time.time()
    # Sample triplanes and return Gaussian params + a,b,lambda
    means3D, rotation, opacity, colors, scales, texsample, texscale, invariance = process_full_Gaussians(pc)
    # t2 = time.time()
    # Precompute point a,b,s texture indexing
    colors_final = []
    for texture, cam in zip(textures, viewpoint_camera):
        dir_pp = (means3D - cam.camera_center.cuda().repeat(colors.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        
        shs_view = colors.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

        shs_view = texsample.transpose(1, 2).view(-1, 2, 16)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        texsample_ab = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        shs_view = invariance.transpose(1, 2).view(-1, invariance.shape[-1], 16)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        tex_invariance = torch.clamp_min(sh2rgb + 0.5, 0.0)
        
        if exposure is not None:
            tex_invariance = exposure * tex_invariance
        
        colors_ibl = sample_mipmap(texture.cuda(), texsample_ab, texscale, num_levels=mip_level)
        color_d = tex_invariance*colors_ibl
        colors_final.append((colors_precomp + color_d).unsqueeze(0))
        
    colors_final = torch.cat(colors_final, dim=0)
    
    # t3 = time.time()
    
    M = len(textures)
    means3D_final = means3D.unsqueeze(0).repeat(M, 1, 1)
    rotation_final = rotation.unsqueeze(0).repeat(M, 1, 1)
    scales_final = scales.unsqueeze(0).repeat(M, 1, 1)
    opacity_final = opacity.unsqueeze(0).repeat(M, 1, 1)

    colors_deform, alpha, meta = rendering_pass(
        means3D_final,
        rotation_final, 
        scales_final, 
        opacity_final, 
        colors_final,
        None, 
        viewpoint_camera, 
        None,
        mode="RGB"
    )

    colors_deform = colors_deform.squeeze(1).permute(0, 3, 1, 2)
    alpha = alpha.squeeze(1).permute(0, 3, 1, 2)
    # print(f"G-call {1./(t2-t1):.4f} MipSamp{1./(t3-t2):.4f} Rend {1./(t4-t3):.4f}")
    
    if return_canon:
        colors_canon, _, _ = rendering_pass(
            means3D, 
            rotation, 
            scales, 
            opacity, 
            colors, 
            None, 
            viewpoint_camera, 
            pc.active_sh_degree,
            mode="RGB"
        )

        colors_canon = colors_canon.squeeze(1).permute(0, 3, 1, 2)

        return colors_deform, colors_canon, alpha, meta
    return colors_deform, None, alpha, meta #(meta, alpha)


def render_canonical(viewpoint_camera, pc):
    """Canonical rendering function

    """
    means3D, rotation, opacity, colors, scales = process_Gaussians(pc)
    
    colors, _, meta = rendering_pass(
        means3D, 
        rotation, 
        scales, 
        opacity, 
        colors,
        None, 
        viewpoint_camera, 
        pc.active_sh_degree,
        mode="RGB"
    )

    return colors.squeeze(0).permute(0, 3, 1, 2), meta



import torch.nn.functional as F
def generate_mipmaps(I, num_levels=3):
    I = I.unsqueeze(0)
    pad_thickness = 1
    maps = [F.pad(I, (pad_thickness, pad_thickness, pad_thickness, pad_thickness), value=0)]    
    for _ in range(1, num_levels):
        # I progressively downsampled
        I = F.interpolate(
            I, scale_factor=0.5,
            mode='bilinear', align_corners=False,
            recompute_scale_factor=True
        )
        # Add zero padding
        I_ = F.pad(I, (pad_thickness, pad_thickness, pad_thickness, pad_thickness), value=0)
        maps.append(I_)
    return maps

def sample_mipmap(I, uv, s, num_levels=3):
    """
    args:
        uv, Tensor, N,2
        s, Tensor, N,1
        I, Tensor, 3, H, W
    """
    N = s.size(0)
    
    # print(I.shape)
    # exit()
    # 1. Generate mipmaps
    maps = generate_mipmaps(I, num_levels=num_levels)
    
    # Normalize us -1, 1 (from 0, 1)
    uv = 2.*uv -1.
    uv = uv.unsqueeze(0).unsqueeze(0) # for grid_sample input we need, N,Hout,Wout,2, where N =1, and W=number of points
    
    # Scaling mip-maps
    L = s*(num_levels-1.)
    lower = torch.floor(L).long().clamp(max=num_levels-1)
    upper = torch.clamp(lower + 1, max=num_levels-1)
    s_interp = (L - lower.float())

    # Initialize mipmap samples
    mip_samples = torch.empty((N, num_levels, 3), device=s.device)    

    # For each map sample using u,v and store the values in samples
    for idx, map in enumerate(maps):
        # map is (1, 3, h, w)
        mip_samples[:, idx] = F.grid_sample(map, uv, mode='bilinear', padding_mode='border', align_corners=False).squeeze(2).squeeze(0).permute(1,0)

    gather_idx_low  = lower.view(N, 1, 1).expand(-1, 1, 3)
    gather_idx_high = upper.view(N, 1, 1).expand(-1, 1, 3)
    colors_low  = torch.gather(mip_samples, 1, gather_idx_low).squeeze(1)   # [N,3]
    colors_high = torch.gather(mip_samples, 1, gather_idx_high).squeeze(1) 
    
    colors = (1. - s_interp) * colors_low + s_interp * colors_high

    return colors

    
