''' Executes EyeLiner Pairwise Registration Pipeline on an Image Dataset '''

# =================
# Install libraries
# =================

import argparse
import logging
import os, sys
import pandas as pd
import torch
from torchvision.transforms import ToPILImage
from data import SequentialDataset
from utils import none_or_str
from eyeliner import EyeLinerP
from visualize import visualize_kp_matches, create_video_from_tensor
from matplotlib import pyplot as plt

def create_logger(log_file_name):
    """Creates a logger object that writes to a specified log file."""
    # Create a logger
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

    # Create a file handler that logs debug and higher level messages
    handler = logging.FileHandler(log_file_name)
    handler.setLevel(logging.DEBUG)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)

    return logger, handler

def parse_args():
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('-d', '--data', default='/sddata/projects/GA_progression_modeling/results/09172024_coris/area_comparisons_af.csv', type=str, help='Dataset csv path')
    parser.add_argument('-m', '--mrn', default='PID', type=str, help='MRN column')
    parser.add_argument('-l', '--lat', default='Laterality', type=str, help='Laterality column')
    parser.add_argument('-sq', '--sequence', default='ExamDate', type=str, help='Sequence ordering column')
    parser.add_argument('-i', '--input', default='file_path_coris', type=str, help='Image column')
    parser.add_argument('-v', '--vessel', default='file_path_vessel_seg', type=none_or_str, help='Vessel column')
    parser.add_argument('-o', '--od', default='file_path_ga_seg', type=none_or_str, help='Disk column')
    parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
    parser.add_argument('--inp', help='Input image to keypoint detector', default='vessel', choices=['img', 'vessel', 'disk', 'peripheral'])

    # keypoint detector args
    parser.add_argument('--reg2start', action='store_true', help='Register all timepoints to the start of the sequence')
    parser.add_argument('--reg_method', help='Registration method', type=str, default='tps')
    parser.add_argument('--lambda_tps', help='TPS lambda parameter', type=float, default=1.)

    # misc
    parser.add_argument('--device', default='cuda:0', help='Device to run program on')
    parser.add_argument('--save', default='results/ga_09172024_coris_af/', help='Location to save results')
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
    logs_folder = os.path.join(args.save, 'logs')
    os.makedirs(reg_matches_save_folder, exist_ok=True)
    os.makedirs(reg_params_save_folder, exist_ok=True)  
    os.makedirs(reg_videos_save_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    for i in range(len(dataset)):
        print(f'Registering patient {i}: ', os.path.join(logs_folder, f'patient_{i}.log'))
        batch_data = dataset[i]

        # create logs file
        logger, handler = create_logger(os.path.join(logs_folder, f'patient_{i}.log'))

        # registration intermediate tensors saved here 
        sequence_registered_images = [batch_data['images'][0]]
        sequence_registered_inputs = [batch_data['inputs'][0]]

        # registration filepaths are saved here
        registration_matches_filenames = [None]
        registration_params_filepaths = [None]
        statuses = ['None']
        
        for j in range(1, len(batch_data['images'])):

            is_registered = False

            # register to the starting point
            if args.reg2start:

                try:
                    logger.info(f"Registering timepoint {j} to 0.")

                    data = {
                        'fixed_input': sequence_registered_inputs[0],
                        'moving_input': batch_data['inputs'][j],
                        'fixed_image': sequence_registered_images[0],
                        'moving_image': batch_data['images'][j],
                    }

                    # compute the registration and save it
                    theta, cache = eyeliner(data)
                    filename = os.path.join(reg_params_save_folder, f'reg_{i}_{j}_0.pth')
                    registration_params_filepaths.append(filename)
                    torch.save(theta, filename)

                    # visualize keypoint matches and save
                    filename = os.path.join(reg_matches_save_folder, f'kp_match_{i}_{j}_0.png')
                    visualize_kp_matches(
                        data['fixed_image'], 
                        data['moving_image'], 
                        cache['kp_fixed'], 
                        cache['kp_moving']
                        )
                    plt.savefig(filename)
                    plt.close()
                    registration_matches_filenames.append(filename)

                    is_registered = True
                    logger.info(f"Successfully registered timepoint {j} to 0.")

                except Exception as e:

                    # get the previous timepoint
                    k = j - 1
                    
                    # log the error
                    logger.error(f"Could not register timepoint {j} to 0. Function failed with error: {e}.")

                    # could not register
                    while True:
                        # don't register if you're back to timepoint 0!
                        if k == 0:
                            is_registered = False
                            registration_params_filepaths.append(None)
                            registration_matches_filenames.append(None)
                            logger.info(f"Saving unregistered image.")
                            break

                        # try to re-register
                        try:
                            logger.info(f"Registering timepoint {j} to {k}.")

                            data = {
                                'fixed_input': sequence_registered_inputs[k],
                                'moving_input': batch_data['inputs'][j],
                                'fixed_image': sequence_registered_images[k],
                                'moving_image': batch_data['images'][j],
                            }

                            # compute registration and save
                            theta, cache = eyeliner(data)
                            filename = os.path.join(reg_params_save_folder, f'reg_{i}_{j}_{k}.pth')
                            registration_params_filepaths.append(filename)
                            torch.save(theta, filename)

                            # visualize keypoint matches and save
                            filename = os.path.join(reg_matches_save_folder, f'kp_match_{i}_{j}_{k}.png')
                            visualize_kp_matches(
                                data['fixed_image'], 
                                data['moving_image'], 
                                cache['kp_fixed'], 
                                cache['kp_moving']
                                )
                            plt.savefig(filename)
                            plt.close()
                            registration_matches_filenames.append(filename)

                            is_registered = True
                            logger.info(f"Successfully registered timepoint {j} to {k}.")
                            break
                        except:
                            is_registered = False
                            logger.error(f"Could not register timepoint {j} to {k}. Function failed with error: {e}.")
                            k = k - 1

            # TODO: register to the previous timepoint
            else:
                k = j - 1
                
                while True:
                    try:
                        logging.info(f"Registering timepoint {j} to {k}.")

                        data = {
                            'fixed_input': sequence_registered_inputs[k],
                            'moving_input': batch_data['inputs'][j],
                            'fixed_image': sequence_registered_images[k],
                            'moving_image': batch_data['images'][j],
                        }

                        # compute the registration and save it
                        theta, cache = eyeliner(data)
                        filename = os.path.join(reg_params_save_folder, f'reg_{i}_{j}_{k}.pth')
                        registration_params_filepaths.append(filename)
                        torch.save(theta, filename)

                        # visualize keypoint matches and save
                        filename = os.path.join(reg_matches_save_folder, f'kp_match_{i}_{j}_{k}.png')
                        visualize_kp_matches(
                            data['fixed_image'], 
                            data['moving_image'], 
                            cache['kp_fixed'], 
                            cache['kp_moving']
                            )
                        plt.savefig(filename)
                        plt.close()
                        registration_matches_filenames.append(filename)

                        is_registered = True
                        logging.info(f"Successfully registered timepoint {j} to {k}.")
                        break

                    except Exception as e:
                        is_registered = False
                        logging.error(f"Could not register timepoint {j} to {k}. Function failed with error: {e}.")
                        if k == 0:
                            registration_params_filepaths.append(None)
                            registration_matches_filenames.append(None)
                            logging.info(f"Saving unregistered image.")
                            break
                        else:
                            k = k - 1

            # create registered image and store for next registration
            if is_registered:

                # apply paramters to image
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
                statuses.append('Pass')
            else:
                sequence_registered_images.append(None)
                sequence_registered_inputs.append(None)
                statuses.append('Fail') 

        # create registration video and save
        sequence_registered_images = torch.stack([im for im in sequence_registered_images if im is not None], dim=0)
        video_save_path = os.path.join(reg_videos_save_folder, f'video{i}.mp4')
        create_video_from_tensor(sequence_registered_images, output_file=video_save_path, frame_rate=10)

        # save registered sequence
        df = batch_data['df']
        df['params'] = registration_params_filepaths
        df['matches'] = registration_matches_filenames
        df['video'] = [video_save_path]*len(df)
        df['logs'] = [os.path.join(logs_folder, f'patient_{i}.log')]*len(df)
        df['status'] = statuses
        results.append(df)

        # Remove handler to prevent duplicate logs in subsequent iterations
        logger.removeHandler(handler)

        # Close the file handler to release the file
        handler.close()
        
    # save results file
    results = pd.concat(results, axis=0, ignore_index=False)
    results.to_csv(os.path.join(args.save, 'results.csv'), index=False)
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)