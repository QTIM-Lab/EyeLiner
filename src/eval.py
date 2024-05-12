''' Evaluates EyeLiner Pairwise Registration Pipeline on an Image Dataset '''

# =================
# Install libraries
# =================

import argparse
import os, sys
from tqdm import tqdm
from utils import none_or_str
import torch
from torchvision.transforms import ToPILImage
from data import ImageDataset
from eyeliner import EyeLinerP
from visualize import create_flicker, create_checkerboard, create_diff_map
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('-d', '--data', default='UCHealth_Annotations/grant_images_pairs_wmasks__.csv', type=str, help='Dataset csv path')
    parser.add_argument('-f', '--fixed', default='fixed_image', type=str, help='Fixed column')
    parser.add_argument('-m', '--moving', default='moving_image', type=str, help='Moving column')
    parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
    parser.add_argument('-r', '--registration', default='registration_path', type=none_or_str, help='Registration column')
    parser.add_argument('-k', '--keypoint', default='keypoints', type=none_or_str, help='Keypoints column')

    # misc
    parser.add_argument('--device', default='cpu', help='Device to run program on')
    parser.add_argument('--save', default='trained_models/', help='Location to save results')
    args = parser.parse_args()
    return args

def main(args):

    device = torch.device(args.device)

    # load dataset
    dataset = ImageDataset(
        path=args.data, 
        input_col=args.fixed, 
        output_col=args.moving,
        input_dim=(args.size, args.size), 
        cmode='rgb',
        input='img',
        keypoints_col=args.keypoint,
        registration_col=args.registration
    )

    # make directory to store registrations
    reg_images_save_folder = os.path.join(args.save, 'registration_images')
    checkerboard_save_folder = os.path.join(args.save, 'ckbd_images')
    flicker_save_folder = os.path.join(args.save, 'flicker_images')
    difference_map_save_folder = os.path.join(args.save, 'diff_map_images')
    os.makedirs(reg_images_save_folder, exist_ok=True)
    os.makedirs(checkerboard_save_folder, exist_ok=True)
    os.makedirs(flicker_save_folder, exist_ok=True)
    os.makedirs(difference_map_save_folder, exist_ok=True)

    images_filenames = []
    ckbd_filenames = []
    flicker_filenames = []
    difference_maps_filenames = []
    if args.keypoint is not None:
        error = []

    for i in tqdm(range(len(dataset))):
        
        # load images
        batch_data = dataset[i]
        fixed_image = batch_data['fixed_image']
        moving_image = batch_data['moving_image']
        theta = batch_data['theta']

        # if image pair could not be registered
        if theta is None:
            images_filenames.append(None)
            ckbd_filenames.append(None)
            flicker_filenames.append(None)
            difference_maps_filenames.append(None)
            continue

        # register moving image
        reg_image = EyeLinerP.apply_transform(theta[1], moving_image.unsqueeze(0)).squeeze(0)
        
        # create mask
        reg_mask = torch.ones_like(moving_image)
        reg_mask = EyeLinerP.apply_transform(theta[1], reg_mask.unsqueeze(0)).squeeze(0)
        
        # apply mask to images
        fixed_image = fixed_image * reg_mask
        moving_image = moving_image * reg_mask
        reg_image = reg_image * reg_mask

        # save registration
        filename = os.path.join(reg_images_save_folder, f'reg_{i}.png')
        ToPILImage()(reg_image).save(filename)
        images_filenames.append(filename)

        # qualitative evaluation
        ckbd = create_checkerboard(fixed_image, reg_image, patch_size=32)
        filename = os.path.join(checkerboard_save_folder, f'ckbd_{i}.png')
        ToPILImage()(ckbd).save(filename)
        ckbd_filenames.append(filename)

        filename = os.path.join(flicker_save_folder, f'flicker_{i}.gif')
        create_flicker(fixed_image, reg_image, output_path=filename)
        flicker_filenames.append(filename)

        filename = os.path.join(difference_map_save_folder, f'diff_map_{i}.png')
        create_diff_map(fixed_image, reg_image, filename)
        difference_maps_filenames.append(filename)

        # quantitative evaluation
        if args.keypoint is not None:
            md = None
            error.append(md)

    dataset.data['registration_path'] = images_filenames
    dataset.data['checkerboard'] = ckbd_filenames
    dataset.data['flicker'] = flicker_filenames
    dataset.data['difference_map'] = difference_maps_filenames
    # add columns to dataframe
    if args.keypoint is not None:
        dataset.data['MD'] = error

        # TODO: compute AUC from error

    # save results
    csv_save = os.path.basename(args.data).split('.')[0] + '_results.csv'
    dataset.data.to_csv(os.path.join(args.save, csv_save), index=False)

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)