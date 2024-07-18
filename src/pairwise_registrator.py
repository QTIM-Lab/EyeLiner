''' Executes EyeLiner Pairwise Registration Pipeline on an Image Dataset '''

# =================
# Install libraries
# =================

import argparse
import os, sys
import pandas as pd
from tqdm import tqdm
import torch
from torchvision.transforms import ToPILImage
from data import ImageDataset
from utils import none_or_str
from eyeliner import EyeLinerP
from visualize import visualize_kp_matches
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('-d', '--data', default='data/GA_progression_modelling/af_images_anne_patients_registration.csv', type=str, help='Dataset csv path')
    parser.add_argument('-f', '--fixed', default='fixed_image', type=str, help='Fixed column')
    parser.add_argument('-m', '--moving', default='moving_image', type=str, help='Moving column')
    parser.add_argument('-fv', '--fixed-vessel', default='fixed_vessel_seg', type=none_or_str, help='Fixed vessel column')
    parser.add_argument('-mv', '--moving-vessel', default='moving_vessel_seg', type=none_or_str, help='Moving vessel column')
    parser.add_argument('-fd', '--fixed-disk', default='fixed_disk', type=none_or_str, help='Fixed disk column')
    parser.add_argument('-md', '--moving-disk', default='moving_disk', type=none_or_str, help='Moving disk column')
    parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
    parser.add_argument('--input', help='Input image to keypoint detector', default='img', choices=['img', 'vessel', 'disk', 'peripheral'])

    # keypoint detector args
    parser.add_argument('--kp_method', help='Keypoint detector method', type=str, default='splg')
    parser.add_argument('--reg_method', help='Registration method', type=str, default='affine')
    parser.add_argument('--lambda_tps', help='TPS lambda parameter', type=float, default=none_or_str)

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
        input=args.input
    )

    # load pipeline
    eyeliner = EyeLinerP(
        kp_method=args.kp_method,
        reg=args.reg_method,
        lambda_tps=args.lambda_tps,
        image_size=(3, args.size, args.size),
        device=device
    )

    # make directory to store registration results data
    reg_params_save_folder = os.path.join(args.save, 'registration_params')
    detected_keypoints_save_folder = os.path.join(args.save, 'detected_keypoints')
    reg_matches_save_folder = os.path.join(args.save, 'registration_keypoint_matches')

    os.makedirs(reg_params_save_folder, exist_ok=True)
    os.makedirs(detected_keypoints_save_folder, exist_ok=True)
    os.makedirs(reg_matches_save_folder, exist_ok=True)

    params_filenames = []
    # images_filenames = []
    kp_filenames = []
    kp_matches_filenames = []
    for i in tqdm(range(len(dataset))):
        # compute registration and save result
        batch_data = dataset[i]
        try:
            theta, cache = eyeliner(batch_data)
        except:
            print(f'Unable to register patient {i}. Skipping.')
            params_filenames.append(None)
            # images_filenames.append(None)
            kp_filenames.append(None)
            kp_matches_filenames.append(None)
            continue

        # save keypoint matches
        filename = os.path.join(reg_matches_save_folder, f'kp_match_{i}.png')
        visualize_kp_matches(
            batch_data['fixed_image'], 
            batch_data['moving_image'], 
            cache['kp_fixed'], 
            cache['kp_moving']
            )
        plt.savefig(filename)
        plt.close()
        kp_matches_filenames.append(filename)

        # save parameters
        filename = os.path.join(reg_params_save_folder, f'reg_{i}.pth')
        torch.save(theta, filename)
        params_filenames.append(filename)

        # save detected keypoints
        filename = os.path.join(detected_keypoints_save_folder, f'kp_set_{i}.csv')
        kp_set = torch.cat([cache['kp_fixed'].squeeze(0), cache['kp_moving'].squeeze(0)], dim=1).cpu().numpy()
        kp_set_df = pd.DataFrame(kp_set)
        kp_set_df.to_csv(filename, index=False)
        kp_filenames.append(filename)
        
    # add column to dataframe and save
    dataset.data['registration_params'] = params_filenames
    dataset.data['registration_matches'] = kp_matches_filenames
    dataset.data['detected_keypoints'] = kp_filenames
    csv_save = os.path.basename(args.data).split('.')[0] + '_results.csv'
    dataset.data.to_csv(os.path.join(args.save, csv_save), index=False)

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)