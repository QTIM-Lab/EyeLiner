''' A module for creating pytorch datasets '''

# import libraries
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Grayscale, Resize, ToTensor

class ImageDataset(Dataset):
    def __init__(self, path, input_col, output_col, input_vessel_col=None, output_vessel_col=None, input_od_col=None, output_od_col=None, input_dim=(512, 512), cmode='rgb', input='img', keypoints_col=None, registration_col=None):
        super(ImageDataset, self).__init__()
        self.path = path
        self.data = pd.read_csv(self.path)
        self.input_col = input_col
        self.output_col = output_col
        self.inputs = self.data[input_col]
        self.outputs = self.data[output_col]

        # auxiliary inputs: vessel masks
        self.input_vessel_col = input_vessel_col
        self.output_vessel_col = output_vessel_col
        self.input_vessels = self.data[input_vessel_col] if input_vessel_col is not None else None
        self.output_vessels = self.data[output_vessel_col] if output_vessel_col is not None else None

        # auxiliary inputs: optic disc masks
        self.input_od_col = input_od_col
        self.output_od_col = output_od_col
        self.input_od = self.data[input_od_col] if input_od_col is not None else None
        self.output_od = self.data[output_od_col] if output_od_col is not None else None

        # evaluation
        self.keypoints_col = keypoints_col
        self.registration_col = registration_col
        self.keypoints = self.data[keypoints_col] if keypoints_col is not None else None
        self.registrations = self.data[registration_col] if registration_col is not None else None

        self.cmode = cmode
        self.input_dim = input_dim if isinstance(input_dim, tuple) else (input_dim, input_dim)
        self.input = input

        print(f'Found {len(self.data)} images.')

    def load_image(self, path):
        x = Image.open(path)
        x = Resize(self.input_dim)(x)
        x = Grayscale()(x) if self.cmode == 'gray' else x
        x = ToTensor()(x)
        return x

    def load_registration(self, path):
        return torch.load(path)
    
    def load_keypoints(self, path):
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = dict()
        
        # grab fixed and moving images
        x, y = self.inputs[index], self.outputs[index]
        x = self.load_image(x)
        y = self.load_image(y)
        data['fixed_image'] = x
        data['moving_image'] = y

        # add registration model inputs
        if self.input == 'img':
            data['fixed_input'] = x
            data['moving_input'] = y

        elif self.input == 'vessel':
            assert self.input_vessels is not None
            assert self.output_vessels is not None
            x_v, y_v = self.input_vessels[index], self.output_vessels[index]
            x_v = self.load_image(x_v)
            y_v = self.load_image(y_v)
            data['fixed_input'] = x_v
            data['moving_input'] = y_v

        elif self.input == 'disk':
            assert self.input_od is not None
            assert self.output_od is not None
            x_d, y_d = self.input_od[index], self.output_od[index]
            x_d = self.load_image(x_d)
            y_d = self.load_image(y_d)
            data['fixed_input'] = x_d
            data['moving_input'] = y_d

        elif self.input == 'peripheral':
            assert self.input_vessels is not None
            assert self.output_vessels is not None
            assert self.input_od is not None
            assert self.output_od is not None 
            x_v, y_v = self.input_vessels[index], self.output_vessels[index]
            x_v = self.load_image(x_v)
            y_v = self.load_image(y_v)
            x_d, y_d = self.input_od[index], self.output_od[index]
            x_d = self.load_image(x_d)
            y_d = self.load_image(y_d)

            # binarize vessel inputs
            fixed_vessel = (x_v > 0.5).float()
            moving_vessel = (y_v > 0.5).float()

            # binarize disk inputs
            fixed_disk_mask = 1 - (x_d > 0.5).float()
            moving_disk_mask = 1 - (y_d > 0.5).float()

            # create structural mask
            x_s = fixed_vessel * fixed_disk_mask
            y_s = moving_vessel * moving_disk_mask

            data['fixed_input'] = x_s
            data['moving_input'] = y_s

        # add registration
        if self.registrations is not None:
            reg = self.registrations[index]
            if isinstance(reg, str):
                data['theta'] = self.load_registration(reg)
            else:
                data['theta'] = None

        # add keypoints
        if self.keypoints is not None:
            kp = self.keypoints[index]
            kp = self.load_keypoints(kp)
            data['fixed_keypoints'] = kp[:, :2]
            data['moving_keypoints'] = kp[:, 2:]

        return data