''' A module for creating pytorch datasets '''

# import libraries
import sys
import random
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    RandAffine,
    EnsureChannelFirst,
    LoadImage,
    ScaleIntensity,
    ToTensor,
    Compose
)
from torchvision.transforms import Grayscale, Resize
from math import pi

class CustomDataset(Dataset):
    ''' Abstract PyTorch class for any Dataset '''
    def __init__(self, path, input_col=None, output_col=None):
        super(CustomDataset, self).__init__()
        self.path = path
        self.input_col = input_col
        self.output_col = output_col
        self.data = pd.read_csv(self.path)
        self.inputs = self.data[self.input_col] if input_col is not None else None
        # self.inputs = self.inputs.apply(lambda x: x.replace('/data/retina_datasets_preprocessed/Dynamic_Cropped/', 'data/retina_datasets/'))
        self.outputs = self.data[self.output_col] if output_col is not None else None
        # self.outputs = self.outputs.apply(lambda x: x.replace('/data/retina_datasets_preprocessed/Dynamic_Cropped/', 'data/retina_datasets/'))

    def __getitem__(self, index):
        x = torch.tensor([self.inputs[index]]).float()
        y = torch.tensor([self.class_mapping[self.outputs[index]]]).float()
        return x, y
    
    def __len__(self):
        return len(self.data)

class ImageDataset(CustomDataset):
    def __init__(self, path, input_col, output_col=None, input_vessel_col=None, output_vessel_col=None, input_od_col=None, output_od_col=None, landmarks_col=None, centroid_xcol=None, centroid_ycol=None, dist_map_col=None, input_dim=(512, 512), cmode='rgb', aug_args=None, intensity_transform=False, concat=False):
        super(ImageDataset, self).__init__(path, input_col, output_col)
        self.path = path
        self.input_dim = input_dim if isinstance(input_dim, tuple) else (input_dim, input_dim)

        # auxiliary inputs: vessel masks
        self.input_vessel_col = input_vessel_col
        self.output_vessel_col = output_vessel_col
        self.input_vessels = self.data[input_vessel_col] if input_vessel_col is not None else None #[None]*len(self.data)
        self.output_vessels = self.data[output_vessel_col] if output_vessel_col is not None else None #[None]*len(self.data)

        # auxiliary inputs: optic disc masks
        self.input_od_col = input_od_col
        self.output_od_col = output_od_col
        self.input_od = self.data[input_od_col] if input_od_col is not None else None #[None]*len(self.data)
        self.output_od = self.data[output_od_col] if output_od_col is not None else None #[None]*len(self.data)

        # auxiliary inputs: distance maps
        self.distance_maps = self.data[dist_map_col] if (dist_map_col is not None) else None #[None]*len(self.data)
        
        # auxiliary inputs: optic disc centroid
        assert ((centroid_xcol is not None) and (centroid_ycol is not None)) or ((centroid_xcol is None) and (centroid_ycol is None)), 'Need to provide both x and y columns of landmarks'
        centroid_x = self.data[centroid_xcol] if (centroid_xcol is not None) else None #[None]*len(self.data)
        centroid_y = self.data[centroid_ycol] if (centroid_xcol is not None) else None #[None]*len(self.data)
        self.centroids = np.concatenate([centroid_x, centroid_y], axis=1) if (centroid_xcol is not None) else None

        # auxiliary inputs: landmarks
        # landmarks = np.loadtxt(self.data[landmarks_col])
        # landmarks = np.split(landmarks, 2, axis=-1)
        
        self.cmode = cmode
        self.intensity_transform = intensity_transform
        self.concat = concat

        if isinstance(aug_args, dict):
            self.aug_args = aug_args
        elif isinstance(aug_args, str):
            with open(aug_args) as f:
                self.aug_args = json.load(f)
        elif aug_args is None:
            self.aug_args = None
        else:
            raise Exception('Augmentations must be provided as dictionary or a json file!')
        
        print(f'Found {len(self.data)} images.')

    def __getitem__(self, index):
        
        # grab inputs and output images (if provided)
        if (self.inputs is not None) and (self.outputs is not None):
            x, y = self.inputs[index], self.outputs[index]
            x = '/sddata/data/retina_datasets/UCHealth_10_images/' + x if 'UCHealth' in x else '/sddata/' + x
            y = '/sddata/data/retina_datasets/UCHealth_10_images/' + y if 'UCHealth' in y else '/sddata/' + y
            
            # get fixed image
            x = LoadImage(image_only=True)(x)
            x = x[:, :, :3] if x.shape[-1] > 3 else x
            x = EnsureChannelFirst()(x)
            x = Resize(self.input_dim, antialias=False)(x)
            x = Grayscale()(x) if self.cmode == 'gray' else x
            
            # get moving image
            y = LoadImage(image_only=True)(y)
            y = y[:, :, :3] if y.shape[-1] > 3 else y
            y = EnsureChannelFirst()(y)
            y = Resize(self.input_dim, antialias=False)(y)
            y = Grayscale()(y) if self.cmode == 'gray' else y
        else:
            x, y = torch.ones(1, *self.input_dim), torch.ones(1, *self.input_dim)

        # get vessel masks, if provided
        if (self.input_vessels is not None) and (self.output_vessels is not None):
            x_v, y_v = self.input_vessels[index], self.output_vessels[index]
            x_v = '/sddata/data/retina_datasets/UCHealth_10_images/' + x_v if 'UCHealth' in x_v else '/sddata/' + x_v # '/sddata'
            y_v = '/sddata/data/retina_datasets/UCHealth_10_images/' + y_v if 'UCHealth' in y_v else '/sddata/' + y_v

            # get fixed label
            x_v = LoadImage(image_only=True)(x_v)
            x_v = EnsureChannelFirst()(x_v)
            x_v = Resize(self.input_dim, antialias=False)(x_v)
            x_v = Grayscale()(x_v) if self.cmode == 'gray' else x_v

            # get moving label
            y_v = LoadImage(image_only=True)(y_v)
            y_v = EnsureChannelFirst()(y_v)
            y_v = Resize(self.input_dim, antialias=False)(y_v)
            y_v = Grayscale()(y_v) if self.cmode == 'gray' else y_v
        else:
            x_v, y_v = torch.ones_like(x), torch.ones_like(y)

        # get optic disc masks, if provided
        if (self.input_od is not None) and (self.output_od is not None):
            x_d, y_d = self.input_od[index], self.output_od[index]
            x_d = '/sddata/data/retina_datasets/UCHealth_10_images/' + x_d if 'UCHealth' in x_d else '/sddata/' + x_d
            y_d = '/sddata/data/retina_datasets/UCHealth_10_images/' + y_d if 'UCHealth' in y_d else '/sddata/' + y_d

            # get fixed label
            x_d = LoadImage(image_only=True)(x_d)
            x_d = EnsureChannelFirst()(x_d)
            x_d = Resize(self.input_dim, antialias=False)(x_d)
            x_d = Grayscale()(x_d) if self.cmode == 'gray' else x_d

            # get moving label
            y_d = LoadImage(image_only=True)(y_d)
            y_d = EnsureChannelFirst()(y_d)
            y_d = Resize(self.input_dim, antialias=False)(y_d)
            y_d = Grayscale()(y_d) if self.cmode == 'gray' else y_d
        else:
            x_d, y_d = torch.ones_like(x), torch.ones_like(y)

        # if using same fixed image as moving image, apply a random affine transform to the image (and it's label, if provided)
        if (self.input_col == self.output_col) and (self.aug_args is not None):
            # monai transform
            tr = RandAffine(
                prob=1,
                **self.aug_args
            )
            seed = random.randint(0, 2**20)
            tr.set_random_state(seed)

            # transform images(+labels)
            if self.concat:
                y = tr(torch.cat([y, y_v, y_d], dim=0)).squeeze(0)
                if self.cmode == 'gray':
                    y, y_v, y_d = y[0, :, :][None, :, :], y[1, :, :][None, :, :], y[2, :, :][None, :, :]
                else:
                    y, y_v, y_d = y[:3, :, :], y[3, :, :][None, :, :], y[4, :, :][None, :, :]
            else:
                y = tr(y)
            tr_matrix = tr.rand_affine_grid.get_transformation_matrix() # A: M -> F
        else:
            tr_matrix = torch.eye(3)

        if self.intensity_transform:
            # apply intensity augmentations RandBiasField(prob=0.7, coeff_range=(0.1, 0.15)), , RandGaussianSmooth(prob=0.3, sigma_x=(1, 2), sigma_y=(1, 2))
            # x = Compose([RandAdjustContrast(prob=0.5, gamma=(0.5, 1.5))])(x)
            # y = Compose([RandAdjustContrast(prob=0.5, gamma=(0.5, 1.5))])(y)
            # x = Compose([RandGaussianSmooth(prob=0.5, sigma_x=(1, 2), sigma_y=(1, 2)), RandBiasField(prob=0.5, coeff_range=(-0.4, 0.4)), RandAdjustContrast(prob=0.5, gamma=(0.75, 1.25))])(x)
            # y = Compose([RandGaussianSmooth(prob=0.5, sigma_x=(1, 2), sigma_y=(1, 2)), RandBiasField(prob=0.5, coeff_range=(-0.4, 0.4)), RandAdjustContrast(prob=0.5, gamma=(0.75, 1.25))])(y)
            pass

        # rescale inputs and convert to tensor
        rescaler = Compose([ScaleIntensity(), ToTensor()])
        x = rescaler(x)
        y = rescaler(y)
        x_v = rescaler(x_v) if x_v.max() != x_v.min() else ToTensor()(x_v)
        y_v = rescaler(y_v) if y_v.max() != y_v.min() else ToTensor()(y_v)
        x_d = rescaler(x_d) if x_d.max() != x_d.min() else ToTensor()(x_d)
        y_d = rescaler(y_d) if y_d.max() != y_d.min() else ToTensor()(y_d)

        if self.concat:
            x = [x] if (self.inputs is not None) else []
            x = x + [x_v] if (self.input_vessels is not None) else x
            x = x + [x_d] if (self.input_od is not None) else x
            y = [y] if (self.outputs is not None) else []
            y = y + [y_v] if (self.output_vessels is not None) else y
            y = y + [y_d] if (self.output_od is not None) else y
            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)

        # load distance maps
        if self.distance_maps is not None:
            d = self.distance_maps[index]
            dist_map = torch.load(d).unsqueeze(0)
            dist_map = torch.permute(dist_map, (0, 2, 1))
        else:
            dist_map = torch.ones_like(x)

        # load landmarks
        if self.centroids is not None:
            centroids = self.centroids[index]
            centroids = torch.from_numpy(centroids)
        else:
            centroids = torch.empty(2)

        out = (
            y,
            x,
            tr_matrix,
            y_v,
            x_v,
            y_d,
            x_d,
            dist_map,
            '/sddata' + self.outputs[index] if self.outputs is not None else '',
            '/sddata' + self.inputs[index] if self.inputs is not None else '',
            '/sddata' + self.output_vessels[index] if self.output_vessels is not None else '',
            '/sddata' + self.input_vessels[index] if self.input_vessels is not None else '', 
            '/sddata' + self.output_od[index] if self.output_od is not None else '',
            '/sddata' + self.input_od[index] if self.input_od is not None else '', 
            centroids
        )

        return out