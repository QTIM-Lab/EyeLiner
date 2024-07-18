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
    def __init__(self, path, input_col, output_col, input_vessel_col=None, output_vessel_col=None, input_od_col=None, output_od_col=None, input_dim=(512, 512), cmode='rgb', input='img', detected_keypoints_col=None, manual_keypoints_col=None, registration_col=None):
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
        self.detected_keypoints_col = detected_keypoints_col
        self.manual_keypoints_col = manual_keypoints_col
        self.registration_col = registration_col
        self.manual_keypoints = self.data[manual_keypoints_col] if manual_keypoints_col is not None else None
        self.detected_keypoints = self.data[detected_keypoints_col] if detected_keypoints_col is not None else None
        self.registrations = self.data[registration_col] if registration_col is not None else None

        self.cmode = cmode
        self.input_dim = input_dim if isinstance(input_dim, tuple) else (input_dim, input_dim)
        self.input = input

        print(f'Found {len(self.data)} images.')

    def load_image(self, path):
        x = Image.open(path).convert('RGB')
        w, h = x.size
        x = Resize(self.input_dim)(x)
        x = Grayscale()(x) if self.cmode == 'gray' else x
        x = ToTensor()(x)
        cache = {'h': h, 'w': w}
        return x, cache

    def load_registration(self, path):
        return torch.load(path)
    
    def load_keypoints(self, path, header='infer'):
        if path.endswith('.txt'):
            kp = pd.read_csv(path, sep=" ", header=header)
        else:    
            kp = pd.read_csv(path, header=header)
        return kp.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data = dict()
        
        # grab fixed and moving images
        x, y = self.inputs[index], self.outputs[index]
        x, x_cache = self.load_image(x)
        y, y_cache = self.load_image(y)
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
            x_v, _ = self.load_image(x_v)
            y_v, _ = self.load_image(y_v)
            data['fixed_input'] = x_v
            data['moving_input'] = y_v

        elif self.input == 'disk':
            assert self.input_od is not None
            assert self.output_od is not None
            x_d, y_d = self.input_od[index], self.output_od[index]
            x_d, _ = self.load_image(x_d)
            y_d, _ = self.load_image(y_d)
            data['fixed_input'] = x_d
            data['moving_input'] = y_d

        elif self.input == 'peripheral':
            assert self.input_vessels is not None
            assert self.output_vessels is not None
            assert self.input_od is not None
            assert self.output_od is not None 
            x_v, y_v = self.input_vessels[index], self.output_vessels[index]
            x_v, _ = self.load_image(x_v)
            y_v, _ = self.load_image(y_v)
            x_d, y_d = self.input_od[index], self.output_od[index]
            x_d, _ = self.load_image(x_d)
            y_d, _ = self.load_image(y_d)

            # binarize vessel inputs
            fixed_vessel = x_v # (x_v > 0.5).float()
            moving_vessel = y_v # (y_v > 0.5).float()

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
                theta = self.load_registration(reg)
                if isinstance(theta, tuple):
                    theta = theta[0].squeeze(0), theta[1].squeeze(0)
                else:
                    theta = theta.squeeze(0)
                data['theta'] = theta
            else:
                data['theta'] = None

        # add detected keypoints
        if self.detected_keypoints is not None:
            kp = self.detected_keypoints[index]
            if isinstance(kp, str):
                kp = self.load_keypoints(kp)
                fixed_kp = kp[:, :2]
                moving_kp = kp[:, 2:]
                data['fixed_keypoints_detected'] = torch.from_numpy(fixed_kp)
                data['moving_keypoints_detected'] = torch.from_numpy(moving_kp)
            else:
                data['fixed_keypoints_detected'] = None
                data['moving_keypoints_detected'] = None

        # add manual keypoints
        if self.manual_keypoints is not None:
            kp = self.manual_keypoints[index]  
            if 'UCHealth' in kp:
                kp = self.load_keypoints(kp)
            else:
                kp = self.load_keypoints(kp, header=None)
            fixed_kp = kp[:, :2]
            moving_kp = kp[:, 2:]

            # normalize keypoints to the image dimension space
            fixed_kp[:, 0] = self.input_dim[1] * fixed_kp[:, 0] / x_cache['w']
            fixed_kp[:, 1] = self.input_dim[0] * fixed_kp[:, 1] / x_cache['h']
            moving_kp[:, 0] = self.input_dim[1] * moving_kp[:, 0] / y_cache['w']
            moving_kp[:, 1] = self.input_dim[0] * moving_kp[:, 1] / y_cache['h']

            data['fixed_keypoints_manual'] = torch.from_numpy(fixed_kp).float()
            data['moving_keypoints_manual'] = torch.from_numpy(moving_kp).float()

        return data
    
class SequentialDataset(Dataset):
    def __init__(self, path, mrn_col, lat_col, sequence_col, input_col, vessel_col=None, od_col=None, input_dim=(512, 512), cmode='rgb', input='img'):
        super(SequentialDataset, self).__init__()
        self.path = path
        self.data = pd.read_csv(self.path)
        self.mrn_col = mrn_col
        self.lat_col = lat_col
        self.sequence_col = sequence_col
        unique_combos = self.data[[mrn_col, lat_col]].drop_duplicates(subset=[mrn_col, lat_col])
        self.unique_combos = list(unique_combos.itertuples(index=False, name=None))

        self.input_col = input_col
        self.vessel_col = vessel_col
        self.od_col = od_col
        self.cmode = cmode
        self.input_dim = input_dim if isinstance(input_dim, tuple) else (input_dim, input_dim)
        self.input = input

    def load_image(self, path):
        x = Image.open(path)
        x = Resize(self.input_dim)(x)
        x = Grayscale()(x) if self.cmode == 'gray' else x
        x = ToTensor()(x)
        return x

    def __len__(self):
        return len(self.unique_combos)

    def __getitem__(self, index):

        data = dict()
        data['images'] = []
        data['inputs'] = []

        # get the patient
        pat, lat = self.unique_combos[index]

        # get the patient data
        mrn_data = self.data[(self.data[self.mrn_col] == pat) & (self.data[self.lat_col] == lat)].sort_values(by=self.sequence_col)
        data['df'] = mrn_data

        for i, row in mrn_data.iterrows():

            image = self.load_image(row[self.input_col])
            data['images'].append(image)

            if self.input == 'img':
                data['inputs'].append(image)

            elif self.input == 'vessel':
                assert self.vessel_col is not None
                vessel = self.load_image(row[self.vessel_col])
                data['inputs'].append(vessel)

            elif self.input == 'disk':
                assert self.od_col is not None
                od = self.load_image(row[self.od_col])
                od = 1 - (od > 0.5).float()
                data['inputs'].append(od)

            elif self.input == 'peripheral':
                assert self.vessel_col is not None
                assert self.od_col is not None
                vessel = self.load_image(row[self.vessel_col])
                od = self.load_image(row[self.od_col])

                # create peripheral mask
                vessel = (vessel > 0.5).float()
                od = 1 - (od > 0.5).float()
                data['inputs'].append(vessel * od)

        return data