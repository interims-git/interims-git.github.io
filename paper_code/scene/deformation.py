import torch
import torch.nn as nn
import torch.nn.init as init
from scene.waveplanes import WavePlaneField

from utils.general_utils import strip_symmetric, build_scaling_rotation

class Deformation(nn.Module):
    def __init__(self, W=256, args=None):
        super(Deformation, self).__init__()
        self.W = W
        self.grid = WavePlaneField(args.bounds, args.target_config)

        self.args = args

        self.ratio=0
        self.create_net()
        
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        # inputs scaling, scalingmod=1.0, rotation
        self.covariance_activation = build_covariance_from_scaling_rotation

        
    def set_aabb(self, xyz_max, xyz_min, grid_type='target'):
        if grid_type=='target':
            self.grid.set_aabb(xyz_max, xyz_min)
  
    
    def create_net(self):
        # Prep features for decoding
        net_size = self.W
        self.spatial_enc = nn.Sequential(nn.Linear(self.grid.feat_dim,net_size))

        self.sample_decoder = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 16*2))
        self.scaling_decoder = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 1))
        self.invariance_decoder = nn.Sequential(nn.ReLU(),nn.Linear(net_size, net_size),nn.ReLU(),nn.Linear(net_size, 3))
    
    def query_spacetime(self, rays_pts_emb):
        space = self.grid(rays_pts_emb[:,:3])
        st = self.spatial_enc(space)
        return st
    
    
    def forward(self,rays_pts_emb):

        dyn_feature = self.query_spacetime(rays_pts_emb)

        samples = torch.sigmoid(self.sample_decoder(dyn_feature)).view(-1, 16, 2)
        invariance = torch.sigmoid(self.invariance_decoder(dyn_feature))
        scaling = torch.sigmoid(self.scaling_decoder(dyn_feature))

        return samples, scaling, invariance
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name and 'background' not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name and 'background' not in name:
                parameter_list.append(param)
        return parameter_list
    

class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width

        self.deformation_net = Deformation(W=net_width,  args=args).cuda()

        self.apply(initialize_weights)

    def forward(self, point):

        return  self.deformation_net(
            point,
        )

    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() 

    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()

    def get_dyn_coefs(self, xyz, scale):
        return self.deformation_net.get_dx_coeffs(xyz, scale)

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
            