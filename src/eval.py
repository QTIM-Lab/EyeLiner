''' Evaluates EyeLiner Pairwise Registration Pipeline on an Image Dataset '''

# =================
# Install libraries
# =================

import argparse
import os, sys
from tqdm import tqdm
from utils import none_or_str, compute_dice
from PIL import Image
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
    parser.add_argument('-fv', '--fixed-vessel', default='fixed_vessel_seg', type=none_or_str, help='Fixed vessel column')
    parser.add_argument('-mv', '--moving-vessel', default='moving_vessel_seg', type=none_or_str, help='Moving vessel column')
    parser.add_argument('-fd', '--fixed-disk', default='fixed_disk', type=none_or_str, help='Fixed disk column')
    parser.add_argument('-md', '--moving-disk', default='moving_disk', type=none_or_str, help='Moving disk column')
    parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
    parser.add_argument('-r', '--registration', default='registration_path', type=none_or_str, help='Registration column')
    parser.add_argument('-l', '--lmbda', default=None, type=float, help='Lambda value for computing the TPS quantitative result')
    parser.add_argument('--detected_keypoints', default='keypoints', type=none_or_str, help='Model Detected Keypoints column')
    parser.add_argument('--manual_keypoints', default='keypoints', type=none_or_str, help='Manually annotated Keypoints column')
    parser.add_argument('--fire_eval', action='store_true', help='Run FIRE evaluation')

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
        input_vessel_col=args.fixed_vessel,
        output_vessel_col=args.moving_vessel,
        input_od_col=args.fixed_disk,
        output_od_col=args.moving_disk,
        input_dim=(args.size, args.size), 
        cmode='rgb',
        input='vessel',
        detected_keypoints_col=args.detected_keypoints,
        manual_keypoints_col=args.manual_keypoints,
        registration_col=args.registration
    )

    # make directory to store registrations
    reg_images_save_folder = os.path.join(args.save, 'registration_images')
    seg_overlaps_save_folder = os.path.join(args.save, 'seg_overlaps')
    checkerboard_after_save_folder = os.path.join(args.save, 'ckbd_images_after')
    checkerboard_before_save_folder = os.path.join(args.save, 'ckbd_images_before')
    flicker_save_folder = os.path.join(args.save, 'flicker_images')
    difference_map_save_folder = os.path.join(args.save, 'diff_map_images')
    os.makedirs(reg_images_save_folder, exist_ok=True)
    os.makedirs(checkerboard_after_save_folder, exist_ok=True)
    os.makedirs(checkerboard_before_save_folder, exist_ok=True)
    os.makedirs(flicker_save_folder, exist_ok=True)
    os.makedirs(difference_map_save_folder, exist_ok=True)
    os.makedirs(seg_overlaps_save_folder, exist_ok=True)

    images_filenames = []
    ckbd_filenames_before = []
    ckbd_filenames_after = []
    segmentation_overlaps = []
    flicker_filenames = []
    difference_maps_filenames = []
    dice_before = []
    dice_after = []
    if args.manual_keypoints is not None:
        error_before = []
        error_after = []

    for i in tqdm(range(len(dataset))):
        
        # load images
        batch_data = dataset[i]
        fixed_image = batch_data['fixed_image'] # (3, 256, 256)
        moving_image = batch_data['moving_image'] # (3, 256, 256)
        fixed_vessel = batch_data['fixed_input'] # (3, 256, 256)
        fixed_vessel = (fixed_vessel > 0.5).float()
        moving_vessel = batch_data['moving_input'] # (3, 256, 256
        moving_vessel = (moving_vessel > 0.5).float()
        theta = batch_data['theta'] # (3, 3) or ((N, 2), (256, 256, 2))

        # if image pair could not be registered
        if theta is None:
            images_filenames.append(None)
            ckbd_filenames_after.append(None)
            ckbd_filenames_before.append(None)
            flicker_filenames.append(None)
            difference_maps_filenames.append(None)
            dice_before.append(None)
            dice_after.append(None)
            if args.manual_keypoints is not None:
                error_before.append(None)
                error_after.append(None)
            continue

        # register moving image
        try:
            reg_image = EyeLinerP.apply_transform(theta, moving_image) # (3, 256, 256)
        except:
            reg_image = EyeLinerP.apply_transform(theta[1], moving_image)

        # register moving segmentation
        try:
            reg_vessel = EyeLinerP.apply_transform(theta, moving_vessel) # (3, 256, 256)
        except:
            reg_vessel = EyeLinerP.apply_transform(theta[1], moving_vessel)
        reg_vessel = (reg_vessel > 0.5).float()

        # create mask
        reg_mask = torch.ones_like(moving_image)
        try:
            reg_mask = EyeLinerP.apply_transform(theta, reg_mask) # (3, 256, 256)
        except:
            reg_mask = EyeLinerP.apply_transform(theta[1], reg_mask)

        # apply mask to images
        fixed_image = fixed_image * reg_mask
        moving_image = moving_image * reg_mask
        reg_image = reg_image * reg_mask

        # apply mask to segmentations
        fixed_vessel = fixed_vessel #* reg_mask
        moving_vessel = moving_vessel #* reg_mask
        reg_vessel = reg_vessel * reg_mask

        # save registration image
        filename = os.path.join(reg_images_save_folder, f'reg_{i}.png')
        ToPILImage()(reg_image).save(filename)
        images_filenames.append(filename)

        # qualitative evaluation

        # segmentation overlap
        seg1 = ToPILImage()(fixed_vessel)
        seg2 = ToPILImage()(reg_vessel)
        seg_overlap = Image.blend(seg1, seg2, alpha=0.5)
        filename = os.path.join(seg_overlaps_save_folder, f'seg_overlap_{i}.png')
        seg_overlap.save(filename)
        segmentation_overlaps.append(filename)

        # checkerboards - before and after registration
        ckbd = create_checkerboard(fixed_image, reg_image, patch_size=32)
        filename = os.path.join(checkerboard_after_save_folder, f'ckbd_{i}.png')
        ToPILImage()(ckbd).save(filename)
        ckbd_filenames_after.append(filename)

        ckbd = create_checkerboard(fixed_image, moving_image, patch_size=32)
        filename = os.path.join(checkerboard_before_save_folder, f'ckbd_{i}.png')
        ToPILImage()(ckbd).save(filename)
        ckbd_filenames_before.append(filename)

        # flicker animation
        filename = os.path.join(flicker_save_folder, f'flicker_{i}.gif')
        create_flicker(fixed_image, reg_image, output_path=filename)
        flicker_filenames.append(filename)

        # subtraction maps
        filename = os.path.join(difference_map_save_folder, f'diff_map_{i}.png')
        create_diff_map(fixed_image, reg_image, filename)
        difference_maps_filenames.append(filename)

        # compute dice between segmentation maps
        seg_dice_before = compute_dice(fixed_vessel, moving_vessel)
        seg_dice_after = compute_dice(fixed_vessel, reg_vessel)
        dice_before.append(seg_dice_before)
        dice_after.append(seg_dice_after)

        # quantitative evaluation
        if args.manual_keypoints is not None:

            fixed_kp_manual = batch_data['fixed_keypoints_manual']
            moving_kp_manual = batch_data['moving_keypoints_manual']
            fixed_kp_detected = batch_data['fixed_keypoints_detected']
            moving_kp_detected = batch_data['moving_keypoints_detected']

            # apply theta to keypoints
            try:
                reg_kp = EyeLinerP.apply_transform_points(theta, moving_keypoints=moving_kp_manual)
            except:                
                reg_kp = EyeLinerP.apply_transform_points(theta[0], moving_keypoints=moving_kp_manual, ctrl_keypoints=moving_kp_detected, tgt_keypoints=fixed_kp_detected, lmbda=args.lmbda)
                # reg_kp = EyeLinerP.apply_transform_points(theta[0], moving_keypoints=fixed_kp_manual, ctrl_keypoints=fixed_kp_detected)

            if args.fire_eval:
                fixed_kp_manual = 2912. * fixed_kp_manual / 256.
                moving_kp_manual = 2912. * moving_kp_manual / 256.
                reg_kp = 2912. * reg_kp / 256.

            # compute mean distance between fixed and registered keypoints
            md_before = torch.sqrt(torch.sum((fixed_kp_manual - moving_kp_manual)**2, dim=-1)).mean().item()
            md_after = torch.sqrt(torch.sum((fixed_kp_manual - reg_kp)**2, dim=-1)).mean().item()
            error_before.append(md_before)
            error_after.append(md_after)

    dataset.data['registration_path'] = images_filenames
    dataset.data['checkerboard_before'] = ckbd_filenames_before
    dataset.data['checkerboard_after'] = ckbd_filenames_after
    dataset.data['flicker'] = flicker_filenames
    dataset.data['seg_overlap'] = segmentation_overlaps
    dataset.data['difference_map'] = difference_maps_filenames
    dataset.data['DICE_before'] = dice_before
    dataset.data['DICE_after'] = dice_after
    # add columns to dataframe
    if args.manual_keypoints is not None:
        dataset.data['MD_before'] = error_before
        dataset.data['MD_after'] = error_after

    # save results
    csv_save = os.path.basename(args.data).split('.')[0] + '_results.csv'
    dataset.data.to_csv(os.path.join(args.save, csv_save), index=False)

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)