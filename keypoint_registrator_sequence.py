''' Patient Sequence Registration API '''

# =================
# Install libraries
# =================
import argparse
import os, sys
sys.path.append(os.getcwd())
from tqdm import tqdm

import torch
import pandas as pd
from PIL import Image
from skimage.exposure import match_histograms
from torchvision.transforms import ToPILImage
from monai.transforms import LoadImage, EnsureChannelFirst, Resize, ScaleIntensity, ToTensor
from src.detectors import get_keypoints
from src.reg_utils import align_img, warp_affine, unnormalize_coordinates, normalize_coordinates, TPS
from lightglue import viz2d
from matplotlib import pyplot as plt
from kornia.geometry.ransac import RANSAC

# =================
# Utility functions
# =================

def none_or_str(value):
    if value == 'None':
        return None
    return value

def load_image(path):
    x = LoadImage(image_only=True)('/sddata/data/' + path)
    x = EnsureChannelFirst()(x)
    x = Resize((256, 256), anti_aliasing=False)(x)
    x = ScaleIntensity()(x)
    x = ToTensor()(x)
    x = torch.permute(x, (0, 2, 1))
    return x.unsqueeze(0)

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

def compute_tps(keypoints_moving, keypoints_fixed, grid_shape, lmbda):
    theta, grid = TPS(dim=2).grid_from_points(keypoints_moving, keypoints_fixed, grid_shape=grid_shape, lmbda=lmbda)   
    return theta, grid

def KPRefiner(keypoints_fixed, keypoints_moving, image_width=256, k=1.5):

    # ============
    # GRAPH METHOD
    # ============

    # # print('Fixed matrix')
    # # get fully-connected graphs with each point being a node and each edge being the length
    # graph_fixed = torch.cdist(keypoints_fixed, keypoints_fixed, p=2)
    # graph_fixed = torch.tril(graph_fixed, diagonal=-1)
    # graph_fixed_flat = graph_fixed[graph_fixed != 0]

    # # print('Moving matrix')
    # graph_moving = torch.cdist(keypoints_moving, keypoints_moving, p=2)
    # graph_moving = torch.tril(graph_moving, diagonal=-1)
    # graph_moving_flat = graph_moving[graph_moving != 0]

    # # Remove scaling factor
    # s = torch.dot(graph_fixed_flat, graph_moving_flat) / torch.dot(graph_moving_flat, graph_moving_flat)
    # graph_moving = s * graph_moving    
    # errors = graph_fixed - graph_moving

    # print(errors)

    # ============
    # RANSAC
    # ============
    _, mask = RANSAC(model_type='homography')(keypoints_fixed.squeeze(0), keypoints_moving.squeeze(0))
    print(mask)
    mask = mask.squeeze()

    keypoints_fixed_filtered = keypoints_fixed[:, mask]
    keypoints_moving_filtered = keypoints_moving[:, mask]
    return keypoints_fixed_filtered, keypoints_moving_filtered

def main(args):
    if os.path.exists(args.save):
        overwrite = input(f'Overwrite existing folder: {args.save}? (y/n/r) ')
        if overwrite == 'y' or overwrite == '':
            from shutil import rmtree
            rmtree(args.save)
        elif overwrite == 'r':
            pass
        else:
            print('Quitting.')
            return
    else:
        os.makedirs(args.save, exist_ok=True)
    device = torch.device(args.device)

    torch.manual_seed(1399)

    # ===========================
    # 1. load patient time series
    # ===========================
    df = pd.read_csv(args.data)

    # ==================
    # 2. Begin inference
    # ==================  
    for pat, pat_df in df.groupby('mrn'):
        if pat < 2088720:
            continue
        # if not pat == 1856804:
            # continue
        # for fileeye, fileeyedf in pat_df.groupby('fileye'):
        for _, fileeyedf in pat_df.iterrows():
            
            print(f'Registering patient {pat} sequence')

            # ===========================
            # 2.1 load up sequence images
            # ===========================

            sequence_unregistered_all_paths = fileeyedf.dropna()
            sequence_unregistered_images_paths = sequence_unregistered_all_paths.filter(like='image').values
            sequence_unregistered_vessels_paths = sequence_unregistered_all_paths.filter(like='vessel').values
            
            sequence_unregistered_images = [load_image(f) for f in sequence_unregistered_images_paths]
            sequence_unregistered_vessels = [load_image(f) for f in sequence_unregistered_vessels_paths]

            # ===============================
            # 2.2 Begin sequence registration
            # ===============================

            sequence_registered_images = [sequence_unregistered_images[0]]
            sequence_registered_vessels = [sequence_unregistered_vessels[0]]
            for j in tqdm(range(1, len(sequence_unregistered_images))):
                
                fixed = sequence_registered_images[-1]
                fixed_vessels = sequence_registered_vessels[-1]

                moving = sequence_unregistered_images[j]
                moving_vessels = sequence_unregistered_vessels[j]
                
                fixed_disks = None
                moving_disks = None

                # =============================
                # 2.2.1 Deep Keypoint Detection
                # =============================
                try:
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
                        mask=args.mask
                        )
                    n = keypoints_fixed.shape[1]
                    if n < 3:
                        print(f'Found {n} matching keypoints between fixed and moving images! Not enough to register!')
                        sequence_registered_images.append(torch.ones_like(fixed))
                        sequence_registered_vessels.append(torch.ones_like(fixed_vessels))
                        continue

                except:
                    print('Error! Not able to register image pair!')
                    sequence_registered_images.append(torch.ones_like(fixed))
                    sequence_registered_vessels.append(torch.ones_like(fixed_vessels))
                    continue

                # ================================
                # 2.2.2 Keypoint Refinement Module
                # ================================
                
                keypoints_fixed, keypoints_moving = KPRefiner(keypoints_fixed, keypoints_moving)
                if keypoints_fixed.shape[1] == 0:
                    print('KP Refinement failed! Cannot register this case!')
                    sequence_registered_images.append(torch.ones_like(fixed))
                    sequence_registered_vessels.append(torch.ones_like(fixed_vessels))
                    continue

                # save keypoint matches
                os.makedirs(os.path.join(args.save, 'keypoint_matches'), exist_ok=True)
                kp_match_save_path = os.path.join(os.path.join(args.save, 'keypoint_matches', f'{j}_' + os.path.basename(sequence_unregistered_images_paths[j])))
                # visualize keypoint correspondences
                axes = viz2d.plot_images([fixed_vessels.squeeze(0), moving_vessels.squeeze(0)])
                viz2d.plot_matches(keypoints_fixed.squeeze(0), keypoints_moving.squeeze(0), color="lime", lw=0.2)
                plt.savefig(kp_match_save_path)
                plt.close()

                # =============================
                # 2.2.2 Registration module
                # =============================
                if args.reg_method == 'affine':
                    A = compute_lq_affine(keypoints_fixed, keypoints_moving).float()
                elif args.reg_method == 'tps':
                    keypoints_fixed = normalize_coordinates(keypoints_fixed, fixed.shape[2:])
                    keypoints_moving = normalize_coordinates(keypoints_moving, moving.shape[2:])
                    theta, grid = compute_tps(keypoints_moving, keypoints_fixed, fixed.shape, torch.tensor(args.lambda_tps))
                    keypoints_fixed = unnormalize_coordinates(keypoints_fixed, fixed.shape[2:])
                    keypoints_moving = unnormalize_coordinates(keypoints_moving, moving.shape[2:])
                else:
                    raise NotImplementedError

                # =============================
                # 2.2.3 Warp moving images
                # =============================
                if args.reg_method == 'affine':
                    registered = warp_affine(moving, A)
                    registered_vessels = warp_affine(moving_vessels, A)
                elif args.reg_method == 'tps':
                    registered = align_img(grid, moving)
                    registered_vessels = align_img(grid, moving_vessels)
                else:
                    raise NotImplementedError
                
                sequence_registered_images.append(registered)
                sequence_registered_vessels.append(registered_vessels)

            # ============================
            # 2.3 Save registered sequence
            # ============================

            assert len(sequence_registered_images) == len(sequence_unregistered_images)
            assert len(sequence_registered_images) == len(sequence_unregistered_vessels)

            os.makedirs(os.path.join(args.save, 'images'), exist_ok=True)
            os.makedirs(os.path.join(args.save, 'vessels'), exist_ok=True)
            for j in range(len(sequence_registered_images)):
                img = ToPILImage()(sequence_registered_images[j].squeeze(0))
                vessel = ToPILImage()(sequence_registered_vessels[j].squeeze(0))
                img.save(os.path.join(args.save, 'images', os.path.basename(sequence_unregistered_images_paths[j])))
                vessel.save(os.path.join(args.save, 'vessels', os.path.basename(sequence_unregistered_vessels_paths[j])))

    # ===================================
    # 3. Modify file names in df and save
    # ===================================
    for col in df.columns:
        if 'image' in col:
            df[col] = df[col].apply(lambda x: os.path.join(args.save, 'images', os.path.basename(x)) if isinstance(x, str) else x)
        elif 'vessel' in col:
            df[col] = df[col].apply(lambda x: os.path.join(args.save, 'vessels', os.path.basename(x)) if isinstance(x, str) else x)
        else:
            df[col] = df[col]

    df.to_csv(os.path.join(args.save, 'results.csv'), index=False)
    return None

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('-d', '--data', default='UCHealth_Annotations/grant_images_pairs_wmasks__.csv', type=str, help='Dataset csv path')
    
    # keypoint detector args
    parser.add_argument('--kp_method', help='Keypoint detection method', choices=['seg', 'superpoint', 'loftr'])
    parser.add_argument('--desc_method', help='Descriptor computation method', choices=['sift', 'superpoint', 'loftr'])
    parser.add_argument('--match_method', help='Descriptor matching method', choices=['lightglue_sift', 'lightglue_superpoint', 'bf', 'flann', 'loftr'])
    parser.add_argument('--reg_method', help='Registration method', type=str, default='affine')
    parser.add_argument('--lambda_tps', help='TPS lambda parameter', type=float, default=0)
    parser.add_argument('--input', help='Input image to keypoint detector', default='img', choices=['img', 'vmask', 'dmask', 'structural'])
    parser.add_argument('--mask', help='Mask out certain predited keypoints', default=None, choices=['vmask', 'dmask', 'structural'])
    
    # others
    parser.add_argument('--save', type=str, default='results_uchealth/', help='Save location for images and csv')
    parser.add_argument('--device', default='cpu', help='Device to run program on')
    args = parser.parse_args()
    
    # run main
    main(args)
    print(f'Results Saved to: {args.save}')