import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class HarveyDataset(Dataset):
    def __init__(self, image_paths, seg_paths,transform = None):
        self.transforms = transform
        self.images,self.masks = [],[]
        
        imgs = sorted(os.listdir(image_paths))
        self.images.extend([os.path.join(image_paths,img) for img in imgs])
        
        masks = sorted(os.listdir(seg_paths))
        self.masks.extend([os.path.join(seg_paths, mask) for mask in masks])

    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        img = np.array(Image.open(self.images[index]))
        mask = np.array(Image.open(self.masks[index]))
        if self.transforms is not None:
            aug = self.transforms(image=img,mask=mask)
            img = aug['image']
            mask = aug['mask']
            mask = torch.max(mask,dim=2)[0]
        return img,mask

class HarveyDatasetSiamese(Dataset):
    def __init__(self, image_paths_pre, image_paths_post, seg_paths, transform = None):
        self.transforms = transform
        self.images_pre, self.images_post, self.masks = [],[],[]
        
        imgs_pre = sorted(os.listdir(image_paths_pre))
        self.images_pre.extend([os.path.join(image_paths_pre,img) for img in imgs_pre])

        imgs_post = sorted(os.listdir(image_paths_post))
        self.images_post.extend([os.path.join(image_paths_post,img) for img in imgs_post])
        
        masks = sorted(os.listdir(seg_paths))
        self.masks.extend([os.path.join(seg_paths, mask) for mask in masks])

    def __len__(self):
        return len(self.images_pre)

    def __getitem__(self,index):
        img_pre = np.array(Image.open(self.images_pre[index]))
        img_post = np.array(Image.open(self.images_post[index]))
        mask = np.array(Image.open(self.masks[index]))

        if self.transforms is not None:
            aug = self.transforms(image=img_pre, image0=img_post, mask=mask)
            img_pre = aug['image'] 
            img_post = aug['image0'] 
            mask = aug['mask']
            mask = torch.max(mask,dim=2)[0]
        return img_pre, img_post, mask

class HarveyDatasetMeta(Dataset):
    def __init__(self, image_paths_pre, meta_paths, seg_paths, transform = None):
        self.transforms = transform
        self.images_pre, self.meta_path, self.masks = [],[],[]
        
        imgs_pre = sorted(os.listdir(image_paths_pre))
        self.images_pre.extend([os.path.join(image_paths_pre,img) for img in imgs_pre])

        meta_path = sorted(os.listdir(meta_paths))
        self.meta_path.extend([os.path.join(meta_paths,img) for img in meta_path])
        
        masks = sorted(os.listdir(seg_paths))
        self.masks.extend([os.path.join(seg_paths, mask) for mask in masks])

    def __len__(self):
        return len(self.images_pre)

    def __getitem__(self, index):
        img_pre = np.array(Image.open(self.images_pre[index]))
        img_meta = np.array(np.load(self.meta_path[index])).transpose(2,1,0)
        mask = np.array(Image.open(self.masks[index]))

        if self.transforms is not None:
            aug = self.transforms(image=img_pre, mask=mask)
            img_pre = aug['image'] 
            mask = aug['mask']
            mask = torch.max(mask,dim=2)[0]
        
        return img_pre, img_meta, mask