''' Executes EyeLiner Pairwise Registration Pipeline on an Image Dataset '''

# =================
# Install libraries
# =================

import argparse
import os, sys
import pandas as pd
import torch
from torchvision.transforms import ToPILImage
from data import SequentialDataset
from utils import none_or_str
from eyeliner import EyeLinerP
from visualize import visualize_kp_matches, create_video_from_tensor
from matplotlib import pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('-d', '--data', default='data/GA_progression_modelling/af_images_one_patient.csv', type=str, help='Dataset csv path')
    parser.add_argument('-m', '--mrn', default='ptid', type=str, help='MRN column')
    parser.add_argument('-l', '--lat', default='fileeye', type=str, help='Laterality column')
    parser.add_argument('-sq', '--sequence', default='exdatetime', type=str, help='Sequence ordering column')
    parser.add_argument('-i', '--input', default='file_path_coris', type=str, help='Image column')
    parser.add_argument('-v', '--vessel', default=None, type=none_or_str, help='Vessel column')
    parser.add_argument('-o', '--od', default='file_path_ga_segmentation', type=none_or_str, help='Disk column')
    parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
    parser.add_argument('--inp', help='Input image to keypoint detector', default='img', choices=['img', 'vessel', 'disk', 'peripheral'])

    # keypoint detector args
    parser.add_argument('--reg2start', action='store_true', help='Register all timepoints to the start of the sequence')
    parser.add_argument('--reg_method', help='Registration method', type=str, default='tps')
    parser.add_argument('--lambda_tps', help='TPS lambda parameter', type=float, default=1.)

    # misc
    parser.add_argument('--device', default='cuda:0', help='Device to run program on')
    parser.add_argument('--save', default='results/ga_5519923/', help='Location to save results')
    args = parser.parse_args()
    return args

def main(args):

    device = torch.device(args.device)

    # load dataset
    dataset = SequentialDataset(
        path=args.data,
        mrn_col=args.mrn,
        lat_col=args.lat,
        sequence_col=args.sequence,
        input_col=args.input,
        vessel_col=args.vessel,
        od_col=args.od,
        input_dim=(args.size, args.size),
        cmode='rgb',
        input=args.inp
    )

    # load pipeline
    eyeliner = EyeLinerP(
        reg=args.reg_method,
        lambda_tps=args.lambda_tps,
        image_size=(3, args.size, args.size),
        device=device
    )

    # make directory and csv to store registration results
    results = []
    reg_matches_save_folder = os.path.join(args.save, 'registration_keypoint_matches')
    reg_params_save_folder = os.path.join(args.save, 'registration_params')
    reg_videos_save_folder = os.path.join(args.save, 'registration_videos')
    os.makedirs(reg_matches_save_folder, exist_ok=True)
    os.makedirs(reg_params_save_folder, exist_ok=True)  
    os.makedirs(reg_videos_save_folder, exist_ok=True)

    for i in range(len(dataset)):
        batch_data = dataset[i]

        # registration intermediate tensors saved here 
        sequence_registered_images = [batch_data['images'][0]]
        sequence_registered_inputs = [batch_data['inputs'][0]]

        # registration filepaths are saved here
        registration_matches_filenames = [None]
        registration_params_filepaths = [None]
        
        for j in range(1, len(batch_data['images'])):

            # setup pair
            if args.reg2start:
                data = {
                    'fixed_input': sequence_registered_inputs[0],
                    'moving_input': batch_data['inputs'][j],
                    'fixed_image': sequence_registered_images[0],
                    'moving_image': batch_data['images'][j],
                }
            else:
                data = {
                    'fixed_input': sequence_registered_inputs[j-1],
                    'moving_input': batch_data['inputs'][j],
                    'fixed_image': sequence_registered_images[j-1],
                    'moving_image': batch_data['images'][j],
                }

            # register pair
            try:

                # compute registration and save
                theta, cache = eyeliner(data)
                filename = os.path.join(reg_params_save_folder, f'reg_{i}_{j}_{j-1}.pth')
                registration_params_filepaths.append(filename)
                torch.save(theta, filename)

                # create registered image and store for next registration
                try:
                    reg_image = eyeliner.apply_transform(theta[1].squeeze(0), data['moving_image'])
                except:
                    reg_image = eyeliner.apply_transform(theta, data['moving_image'])

                try:
                    reg_input = eyeliner.apply_transform(theta[1].squeeze(0), data['moving_input'])
                except:
                    reg_input = eyeliner.apply_transform(theta, data['moving_input'])
                
                sequence_registered_images.append(reg_image)
                sequence_registered_inputs.append(reg_input)

                # visualize keypoint matches and save
                filename = os.path.join(reg_matches_save_folder, f'kp_match_{i}_{j}_{j-1}.png')
                visualize_kp_matches(
                    data['fixed_image'], 
                    data['moving_image'], 
                    cache['kp_fixed'], 
                    cache['kp_moving']
                    )
                plt.savefig(filename)
                plt.close()
                registration_matches_filenames.append(filename)

            except:
                registration_params_filepaths += [None]*(len(batch_data['images'])-len(sequence_registered_images))
                registration_matches_filenames += [None]*(len(batch_data['images'])-len(sequence_registered_images))
                sequence_registered_images += [torch.zeros(3, args.size, args.size)]*(len(batch_data['images'])-len(sequence_registered_images))
                sequence_registered_inputs += [torch.zeros(3, args.size, args.size)]*(len(batch_data['images'])-len(sequence_registered_inputs))    

                if args.reg2start:
                    print(f'Unable to register images {j} to {0} in patient {i}. Skipping.')
                else:
                    print(f'Unable to register images {j} to {j-1} in patient {i}. Skipping.')
                break

        # create registration video and save
        sequence_registered_images = torch.stack(sequence_registered_images, dim=0)
        video_save_path = os.path.join(reg_videos_save_folder, f'video{i}.mp4')
        create_video_from_tensor(sequence_registered_images, output_file=video_save_path, frame_rate=10)

        # save registered sequence
        df = batch_data['df']
        df['params'] = registration_params_filepaths
        df['matches'] = registration_matches_filenames
        df['video'] = [video_save_path]*len(df)
        results.append(df)
        
    # save results file
    results = pd.concat(results, axis=0, ignore_index=False)
    results.to_csv(os.path.join(args.save, 'results.csv'), index=False)
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)