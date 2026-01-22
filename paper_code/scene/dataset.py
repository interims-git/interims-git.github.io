from torch.utils.data import Dataset
from scene.cameras import Camera
import torch

from torchvision import transforms as T
class FourDGSdataset(Dataset):
    def __init__(
        self,
        dataset,
        dataset_type,
        dataset_name
    ):
        self.dataset = dataset
        self.subset_idxs = None #[len(dataset)]
        if dataset_name == "vsr" and dataset_type == 'test':
            self.dataset = dataset[0] + dataset[1] + dataset[2]
            self.subset_idxs = [len(dataset[0]), len(dataset[1]), len(dataset[2])] # Also the starting index of the following dataset-subset
        
        self.dataset_type = dataset_type
        
        self.loading_flags = {
            "image":True if dataset_type not in ['video'] else False,
            "canon":True if dataset_type in ['train'] else False,
            "scene_occluded":True if dataset_type not in ['video'] else False,
            "differences":False,
        }
        
    def __getitem__(self, index):
        dataset = self.dataset
            
        cam = Camera(
            R=dataset[index].R, T=dataset[index].T,
            fx=dataset[index].fx, fy=dataset[index].fy,
            cx=dataset[index].cx, cy=dataset[index].cy,
            k1=dataset[index].k1, k2=dataset[index].k2,
            p1=dataset[index].p1, p2=dataset[index].p2,
            
            width=dataset[index].width, height=dataset[index].height,

            time=dataset[index].time,

            image_path=dataset[index].image_path,
            sceneoccluded_path=dataset[index].so_path,
            canon_path=dataset[index].canon_path,
            
            uid=dataset[index].uid,
            data_device=torch.device("cuda"), 
        )
        
        if self.loading_flags["image"]:
            if dataset[index].image is None:
                cam.load_image_from_flags("image")
            else:
                cam.image = dataset[index].image
            
        if self.loading_flags["canon"]:
            if dataset[index].canon is None:
                cam.load_image_from_flags("canon")
            else:
                cam.canon = dataset[index].canon
            
        if self.loading_flags["scene_occluded"]:
            if dataset[index].mask is None:
                cam.load_image_from_flags("scene_occluded")
            else:
                cam.sceneoccluded_mask = dataset[index].mask
        return cam

    def __len__(self):
        
        return len(self.dataset)

from PIL import Image
class IBLBackround(Dataset):
    def __init__(
        self,
        dataset,
    ):
        self.dataset = dataset
        self.transform = T.ToTensor()

    def load_custom_image(self, img_path):
        
        image = self.transform(
            Image.open(img_path).convert("RGB")
        )
        return image
        
    def __getitem__(self, index):
        image = self.transform(
            Image.open(self.dataset[index]).convert("RGB")
        )
        return image

    def __len__(self):
        
        return len(self.dataset)

class ABC:
    def __init__(self, abc):
        self.abc = abc
    