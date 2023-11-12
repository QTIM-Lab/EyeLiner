''' Register two images using the kornia registration API '''

# =================
# Install libraries
# =================

import time
from statistics import mean

import argparse
import os, sys
sys.path.append(os.getcwd())
from tqdm import tqdm

import pandas as pd
import numpy as np
from skimage.exposure import match_histograms
import cv2
import random
from PIL import Image, ImageDraw

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, Grayscale

from src.data import ImageDataset
from src.visualize import create_checkerboard

from src.detectors import get_keypoints
from lightglue import viz2d
from matplotlib import pyplot as plt

# =================
# Utility functions
# =================

def none_or_str(value):
    if value == 'None':
        return None
    return value

def draw_coordinates(img, coordinates_tensor, marker_size=2, shape='o', seed=1399):    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Convert torch tensor to a list of tuples
    coordinates = [(int(coord[0]), int(coord[1])) for coord in coordinates_tensor]

    random.seed(seed)
    
    # Draw filled circles at the coordinates on the image
    for coord in coordinates:
        x, y = coord

        # Generate a random RGB color
        marker_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Draw a plus shape centered at the coordinate
        if shape == '+':
            plus_size = 5
            draw.line([(x - plus_size, y), (x + plus_size, y)], fill=marker_color, width=2)
            draw.line([(x, y - plus_size), (x, y + plus_size)], fill=marker_color, width=2)
        else:
            circle_radius = marker_size
            draw.ellipse([x - circle_radius, y - circle_radius, x + circle_radius, y + circle_radius], fill=marker_color)
    
    # Save the modified image
    return img

def get_steve_kp(ids, data='sigf'):
    if data == 'sigf':
        fname = [f'data/retina_datasets/SIGF/SIGF_Annotations/landmarks/{i}.csv' for i in ids]
        lm = [torch.tensor(pd.read_csv(f, header=None).values) for f in fname]
    elif data == 'uchealth':
        fname = [f'data/retina_datasets/UCHealth_10_images/UCHealth_Annotations/landmarks/{i}.csv' for i in ids]
        lm = [torch.tensor(pd.read_csv(f).values) for f in fname]
    else:
        fname = [f'data/retina_datasets/FIRE/Ground Truth/control_points_{i}_1_2.txt' for i in ids]
        lm = [torch.from_numpy(np.genfromtxt(f)) * (256 / 2912) for f in fname]
    lm = torch.stack(lm, dim=0)
    points1, points2 = lm[:, :, :2], lm[:, :, 2:]
    return points1, points2

def histmatch(src, ref):
    matched = match_histograms(src.numpy(), ref.numpy(), channel_axis=0)
    return torch.tensor(matched)

def compute_lq_affine(points0, points1):
    ''' Find the least squares matrix that maps points0 to points1 '''

    # convert to homogenous coordinates
    P = torch.cat([points0, torch.ones(1, points0.shape[1], 1)], dim=2) # (b, n, 3)
    P = torch.permute(P, (0, 2, 1)) # (b, 3, n)
    Q = torch.cat([points1, torch.ones(1, points1.shape[1], 1)], dim=2) # (b, n, 3)
    Q = torch.permute(Q, (0, 2, 1)) # (b, 3, n)

    # compute lq sol
    Q_T = torch.permute(Q, (0, 2, 1)) # (b, n, 3)
    QQ_T = torch.einsum('bmj,bjn->bmn', Q, Q_T) # (b, 3, 3)

    try:
        A = P @ Q_T @ torch.linalg.inv(QQ_T)
    except:
        A = P @ Q_T @ torch.linalg.pinv(QQ_T)

    return A

def warp_affine(moving, A):
    # return AffineTransform(zero_centered=False)(moving, A)

    warped = []
    for i in range(moving.shape[0]):
        warped_image = torch.permute(moving[i], (1, 2, 0)).numpy() # (h, w, c)
        affine_mat = A[i].numpy() # (3, 3)
        warped_image = cv2.warpAffine(warped_image, affine_mat[:2, :], (warped_image.shape[0], warped_image.shape[1]))
        if len(warped_image.shape) < 3:
            warped_image = warped_image[:, :, None]
        warped_image = torch.permute(torch.tensor(warped_image), (2, 0, 1))
        warped.append(warped_image)
    warped_images = torch.stack(warped, dim=0)
    return warped_images

# =============
# Main function
# =============

def main(args):

    timings = []

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
        cmode='rgb'
    )

    # convert to dataloader
    dataloader = DataLoader(dataset, shuffle=False, num_workers=4, batch_size=1)

    # make results folder and df
    os.makedirs(os.path.join(args.save, 'fixed'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'moving'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'registered'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'fixed_vessels'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'registered_vessels'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'fixed_disks'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'registered_disks'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'checkerboards_before'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'checkerboards_after'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'fixed_lm'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'moving_lm'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'registered_lm'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'keypoint_matches'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'keypoint_matches_unfiltered'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'difference_maps'), exist_ok=True)
    results_df = pd.DataFrame(columns=['Fixed', 'Registered', 'Fixed_Vessels', 'Registered_Vessels', 'Fixed_Disks', 'Registered_Disks', 'Difference Map'])
    fixed_images = []
    reg_images = []
    fixed_vessels_images = []
    reg_vessels_images = []
    fixed_disks_images = []
    reg_disks_images = []
    checkerboard_images = []
    registration_error_fixed_moving = []
    registration_error_fixed_reg = []
    diff_maps = []

    # compute the keypoint locations
    step = 0
    for batch_data in tqdm(dataloader):

        # get images
        fixed = batch_data[1]
        fixed = torch.permute(fixed, (0, 1, 3, 2))
        fixed_paths = batch_data[9]
        moving = batch_data[0]
        moving = torch.permute(moving, (0, 1, 3, 2))
        moving_corrected = torch.stack([histmatch(moving[i], fixed[i]) for i in range(fixed.shape[0])], dim=0)
        moving_paths = batch_data[8]
        fixed_vessels = batch_data[4]
        fixed_vessels = torch.permute(fixed_vessels, (0, 1, 3, 2))
        moving_vessels = batch_data[3]
        moving_vessels = torch.permute(moving_vessels, (0, 1, 3, 2))
        fixed_disks = batch_data[6]
        fixed_disks = torch.permute(fixed_disks, (0, 1, 3, 2))
        moving_disks = batch_data[5]
        moving_disks = torch.permute(moving_disks, (0, 1, 3, 2))

        b = fixed.shape[0]

        # if args.landmarks == 'fire':
        #     ids = [f.split('/')[-1].split('_')[0] for f in fixed_paths]
        # else:
        #     ids = [f.split('/')[-2] for f in fixed_paths]
        # keypoints_fixed, keypoints_moving = get_steve_kp(ids, args.landmarks) # (b, n, 2)
        # keypoints_fixed, keypoints_moving = keypoints_fixed.float(), keypoints_moving.float()
        # keypoints_fixed_unfiltered, keypoints_moving_unfiltered = keypoints_fixed, keypoints_moving

        keypoints_fixed, keypoints_moving, keypoints_fixed_unfiltered, keypoints_moving_unfiltered, timing_detection = get_keypoints(
            fixed,
            moving,
            fixed_vessels,
            fixed_disks,
            moving_vessels,
            moving_disks,
            kp_method=args.kp_method,
            desc_method=args.desc_method,
            match_method=args.match_method,
            device=device,
            inp=args.input,
            mask=args.mask,
            top100=args.top_100
        )

        for i in range(b):
            kp_match_save_path = os.path.join(os.path.join(args.save, 'keypoint_matches'), f'{step}_' + os.path.basename(moving_paths[i]).split('.')[0] + '.png')
            # visualize keypoint correspondences
            axes = viz2d.plot_images([fixed.squeeze(0), moving.squeeze(0)])
            viz2d.plot_matches(keypoints_fixed.squeeze(0), keypoints_moving.squeeze(0), color="lime", lw=0.2)
            plt.savefig(kp_match_save_path)

            kp_match_save_path = os.path.join(os.path.join(args.save, 'keypoint_matches_unfiltered'), f'{step}_' + os.path.basename(moving_paths[i]).split('.')[0] + '.png')
            # visualize keypoint correspondences
            axes = viz2d.plot_images([fixed.squeeze(0), moving.squeeze(0)])
            viz2d.plot_matches(keypoints_fixed_unfiltered.squeeze(0), keypoints_moving_unfiltered.squeeze(0), color="lime", lw=0.2)
            plt.savefig(kp_match_save_path)
            plt.close()

        # Start the timer for this iteration
        start_time = time.time()
        # compute least squares affine using key points
        A = compute_lq_affine(keypoints_fixed, keypoints_moving).float()
        # Start the timer for this iteration
        end_time = time.time()

        timing_registration = end_time - start_time
        
        # Put together all the timings
        runtime = sum(timing_detection + [timing_registration])
        timings.append(runtime)

        print(f'Runtime per image: {runtime}s')

        # use Steve's landmarks for evaluation (if not already using his)
        if args.evaluate:
            if args.landmarks == 'fire':
                ids = [f.split('/')[-1].split('_')[0] for f in fixed_paths]
            else:
                ids = [f.split('/')[-2] for f in fixed_paths]
            keypoints_fixed, keypoints_moving = get_steve_kp(ids, args.landmarks) # (b, n, 2)
            keypoints_fixed, keypoints_moving = keypoints_fixed.float(), keypoints_moving.float()

        # compute predictions
        keypoints_moving_ = torch.cat([keypoints_moving, torch.ones(1, keypoints_moving.shape[1], 1)], dim=2)
        keypoints_moving_ = torch.permute(keypoints_moving_, (0, 2, 1)).float()
        keypoints_registered = torch.bmm(A[:, :2, :], keypoints_moving_) # (b, 2, n)
        keypoints_registered = torch.permute(keypoints_registered, (0, 2, 1)) # (b, n, 2)

        # post-process keypoints?
        if args.evaluate:
            if args.landmarks == 'fire':
                keypoints_fixed = keypoints_fixed * (2912 / 256)
                keypoints_moving = keypoints_moving * (2912 / 256)
                keypoints_registered = keypoints_registered * (2912 / 256)

        # compute the error
        mse_fm = torch.mean(torch.abs(keypoints_fixed - keypoints_moving), dim=(-2, -1))
        mse_fr = torch.mean(torch.abs(keypoints_fixed - keypoints_registered), dim=(-2, -1))
        registration_error_fixed_moving.append(torch.mean(mse_fm).item())
        registration_error_fixed_reg.append(torch.mean(mse_fr).item())

        # warp moving images using affine matrix
        mask = warp_affine(torch.ones_like(fixed), A)
        registered = warp_affine(moving, A)
        registered_corrected = warp_affine(moving_corrected, A)
        registered_vessels = warp_affine(moving_vessels, A)
        registered_disks = warp_affine(moving_disks, A)

        registered = registered * mask
        registered_vessels = registered_vessels * mask
        registered_disks = registered_disks * mask
        # fixed = fixed * mask
        # moving = moving * mask

        if registered.shape[1] == 3:
            reg_gray = Grayscale()(registered)
        if moving.shape[1] == 3:
            fixed_gray = Grayscale()(fixed*mask)
        diff_map = reg_gray - fixed_gray #torch.abs(reg_gray - fixed_gray)
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min())

        # save images
        for i in range(b):
        
            fixed_save_path = os.path.join(os.path.join(args.save, 'fixed'), f'{step}_' + os.path.basename(fixed_paths[i]).split('.')[0] + '.png')
            moving_save_path = os.path.join(os.path.join(args.save, 'moving'), f'{step}_' + os.path.basename(moving_paths[i]).split('.')[0] + '.png')
            reg_save_path = os.path.join(os.path.join(args.save, 'registered'), f'{step}_' + os.path.basename(moving_paths[i]).split('.')[0] + '.png')
            cb_before_save_path = os.path.join(os.path.join(args.save, 'checkerboards_before'), f'{step}_'+ os.path.basename(moving_paths[i]).split('.')[0] + '.png')
            cb_after_save_path = os.path.join(os.path.join(args.save, 'checkerboards_after'), f'{step}_'+ os.path.basename(moving_paths[i]).split('.')[0] + '.png')
            fixed_vessels_save_path = os.path.join(os.path.join(args.save, 'fixed_vessels'), f'{step}_' + os.path.basename(fixed_paths[i]).split('.')[0] + '.png')
            reg_vessels_save_path = os.path.join(os.path.join(args.save, 'registered_vessels'), f'{step}_' + os.path.basename(moving_paths[i]).split('.')[0] + '.png')
            fixed_disks_save_path = os.path.join(os.path.join(args.save, 'fixed_disks'), f'{step}_' + os.path.basename(fixed_paths[i]).split('.')[0] + '.png')
            reg_disks_save_path = os.path.join(os.path.join(args.save, 'registered_disks'), f'{step}_' + os.path.basename(moving_paths[i]).split('.')[0] + '.png')
            diff_maps_save_path = os.path.join(os.path.join(args.save, 'difference_maps'), f'{step}_' + os.path.basename(moving_paths[i]).split('.')[0] + '.png')

            # extra paths
            reg_w_lm_save_path = os.path.join(os.path.join(args.save, 'registered_lm'), f'{step}_' + os.path.basename(moving_paths[i]).split('.')[0] + '.png')
            fixed_w_lm_save_path = os.path.join(os.path.join(args.save, 'fixed_lm'), f'{step}_' + os.path.basename(fixed_paths[i]).split('.')[0] + '.png')
            moving_w_lm_save_path = os.path.join(os.path.join(args.save, 'moving_lm'), f'{step}_' + os.path.basename(moving_paths[i]).split('.')[0] + '.png')

            # save checkerboards
            cb = create_checkerboard(fixed[i]*mask[i], registered_corrected[i], patch_size=32)
            cb = ToPILImage()(cb)
            cb.save(cb_after_save_path)

            cb = create_checkerboard(fixed[i]*mask[i], moving_corrected[i]*mask[i], patch_size=32)
            cb = ToPILImage()(cb)
            cb.save(cb_before_save_path)

            # save reg image
            reg = ToPILImage()(registered[i])
            reg.save(reg_save_path)

            # TODO
            # save reg image (with landmarks annotated)
            reg = draw_coordinates(reg, keypoints_registered[i], shape='+')
            reg.save(reg_w_lm_save_path)

            # save fixed image
            fixed = ToPILImage()(fixed[i])
            fixed.save(fixed_save_path)

            # save fixed image
            moving = ToPILImage()(moving[i])
            moving.save(moving_save_path)

            # TODO
            # save fixed image (with landmarks annotated)
            fixed = draw_coordinates(fixed, keypoints_fixed[i], shape='+')
            fixed.save(fixed_w_lm_save_path)

            # TODO
            # save moving image (with landmarks annotated)
            moving = draw_coordinates(moving, keypoints_moving[i], shape='+')
            moving.save(moving_w_lm_save_path)

            # save reg vessels image
            reg = ToPILImage()(registered_vessels[i])
            reg.save(reg_vessels_save_path)

            # save fixed vessels image
            fixed = ToPILImage()(fixed_vessels[i])
            fixed.save(fixed_vessels_save_path)

            # save reg vessels image
            reg = ToPILImage()(registered_disks[i])
            reg.save(reg_disks_save_path)

            # save fixed vessels image
            fixed = ToPILImage()(fixed_disks[i])
            fixed.save(fixed_disks_save_path)

            # save difference maps
            # diff = ToPILImage()(diff_map[i])
            # diff = cv2.applyColorMap(np.uint8(diff_map[i][0].numpy()*255), cv2.COLORMAP_JET)
            # cv2.imwrite(diff_maps_save_path, diff)
            plt.imsave(diff_maps_save_path, diff_map[i][0].numpy(), cmap='viridis')
            plt.close()
            # diff.save(diff_maps_save_path)

            fixed_images.append(fixed_save_path)
            reg_images.append(reg_save_path)
            checkerboard_images.append(cb_after_save_path)
            fixed_disks_images.append(fixed_disks_save_path)
            reg_disks_images.append(reg_disks_save_path)
            fixed_vessels_images.append(fixed_vessels_save_path)
            reg_vessels_images.append(reg_vessels_save_path)
            diff_maps.append(diff_maps_save_path)

            step += 1

    # save to df
    results_df['Fixed'] = fixed_images
    results_df['Registered'] = reg_images
    results_df['Fixed_Vessels'] = fixed_vessels_images
    results_df['Registered_Vessels'] = reg_vessels_images
    results_df['Fixed_Disks'] = fixed_disks_images
    results_df['Registered_Disks'] = reg_disks_images
    results_df['TRE_fixed_moving'] = registration_error_fixed_moving
    results_df['TRE_fixed_registered'] = registration_error_fixed_reg
    results_df.to_csv(os.path.join(args.save, 'results.csv'), index=False)

    print(f'Average runtime per image: {mean(timings)}s')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('-d', '--data', default='UCHealth_Annotations/grant_images_pairs_wmasks__.csv', type=str, help='Dataset csv path')
    parser.add_argument('-f', '--fixed', default='fixed_image', type=str, help='Fixed column')
    parser.add_argument('-m', '--moving', default='moving_image', type=str, help='Moving column')
    parser.add_argument('-fv', '--fixed-vessel', default='fixed_mask', type=none_or_str, help='Fixed vessel column')
    parser.add_argument('-mv', '--moving-vessel', default='moving_mask', type=none_or_str, help='Moving vessel column')
    parser.add_argument('-fd', '--fixed-disk', default='fixed_disk_path', type=none_or_str, help='Fixed disk column')
    parser.add_argument('-md', '--moving-disk', default='moving_disk_path', type=none_or_str, help='Moving disk column')
    
    # keypoint detector args
    parser.add_argument('--kp_method', help='Keypoint detection method', choices=['seg', 'superpoint', 'loftr'])
    parser.add_argument('--desc_method', help='Descriptor computation method', choices=['sift', 'superpoint', 'loftr'])
    parser.add_argument('--match_method', help='Descriptor matching method', choices=['lightglue_sift', 'lightglue_superpoint', 'bf', 'flann', 'loftr'])
    parser.add_argument('--input', help='Input image to keypoint detector', default='img', choices=['img', 'vmask', 'dmask'])
    parser.add_argument('--mask', help='Mask out certain predited keypoints', default=None, choices=['vmask', 'dmask', 'structural'])
    parser.add_argument('--top_100', help='Select only top 100 confident keypoint matches', action='store_true')

    # visualize args
    parser.add_argument('-l', '--landmarks', help='Ground Truth Landmarks source', default=None, type=str)
    
    # others
    parser.add_argument('-e', '--evaluate', help='Flag for whether to compute landmark error or not.', action='store_true')
    parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
    parser.add_argument('--save', type=str, default='results_uchealth/', help='Save location for images and csv')
    parser.add_argument('--device', default='cpu', help='Device to run program on')
    args = parser.parse_args()
    
    main(args)