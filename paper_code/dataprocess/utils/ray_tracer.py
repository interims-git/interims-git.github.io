import torch
import torch.nn as nn

def fov2_f_c(fovx, fovy, H, W):
    fx = 0.5 * W / torch.tan(0.5 * fovx)
    fy = 0.5 * H / torch.tan(0.5 * fovy)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    
    return fx, fy, cx, cy

class SharedIntrinsics(nn.Module):
    
    def __init__(self, H, W, intr, device="cuda"):
        super().__init__()
        
        self.focals = nn.Parameter(torch.tensor([intr[0,0], intr[1,1], intr[0,2], intr[1,2]]).to(device))

        self.H=H
        self.W=W
    
    def forward(self):
        return self.H,self.W, self.focals[0], self.focals[1],self.focals[2],self.focals[3]
        

class TrainCam(nn.Module):
    def __init__(self, c2w, device="cude"):
        super().__init__()
        
        self.device = device
        self.c2w = nn.Parameter(c2w.cuda().float())

    def forward(self):
        return self.c2w

class TrainABC(nn.Module):
    def __init__(self, abcd, H, W, device="cuda", background_texture=None):
        super().__init__()
        self.abc = nn.Parameter(abcd[[3,2,0]].to(device))
        self.H, self.W = H, W
        self.background_texture = background_texture
    
    def forward(self):
        return self.abc 

