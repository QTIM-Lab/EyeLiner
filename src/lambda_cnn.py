import os, json
import sys
sys.path.append(os.getcwd())

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

from monai.transforms import LoadImage, EnsureChannelFirst, Resize, ScaleIntensity, ToTensor
from monai.losses import DiceLoss
from torchvision.transforms import ToPILImage

import pandas as pd
from matplotlib import pyplot as plt

from tqdm import tqdm
from src.reg_utils import normalize_coordinates, unnormalize_coordinates, TPS, align_img
from src.visualize import create_checkerboard
from lightglue import viz2d

def none_or_str(value):
    if value == 'None':
        return None
    return value

def compute_tps(keypoints_moving, keypoints_fixed, grid_shape, lmbda):
    theta, grid = TPS(dim=2).grid_from_points(keypoints_moving, keypoints_fixed, grid_shape=grid_shape, lmbda=lmbda)   
    return theta, grid

class ImageDataset(Dataset):
    def __init__(self, path, fixed_img_col, moving_img_col, fixed_vessel_col, moving_vessel_col, lm='fire'):
        super(ImageDataset, self).__init__()
        self.path = path
        self.fixed_img_col = fixed_img_col
        self.moving_img_col = moving_img_col
        self.fixed_vessel_col = fixed_vessel_col
        self.moving_vessel_col = moving_vessel_col

        # load data
        self.data = pd.read_csv(self.path)
        self.fixed = self.data[self.fixed_img_col] if fixed_img_col is not None else None
        self.moving = self.data[self.moving_img_col] if moving_img_col is not None else None
        self.fixed_vessels = self.data[self.fixed_vessel_col] if fixed_vessel_col is not None else None
        self.moving_vessels = self.data[self.moving_vessel_col] if moving_vessel_col is not None else None
        self.landmarks = lm

    def load_image(self, im):
        x = LoadImage(image_only=True)('/sddata' + im)
        x = x[:, :, :3] if x.shape[-1] > 3 else x
        x = EnsureChannelFirst()(x)
        x = Resize((256, 256), anti_aliasing=False)(x)
        x = ScaleIntensity()(x)
        x = ToTensor()(x)
        x = torch.permute(x, (0, 2, 1))            
        return x
    
    def load_kp(self, ids, data='sigf'):
        if data == 'sigf':
            fname = f'data/retina_datasets/SIGF_time_variant_sequences/landmarks/{ids}.csv'
            lm = torch.tensor(pd.read_csv(fname, header=None).values)
        elif data == 'uchealth':
            fname = f'data/retina_datasets/UCHealth_Annotations/landmarks/{ids}.csv'
            lm = torch.tensor(pd.read_csv(fname).values)
        else:
            fname = f'data/retina_datasets/FIRE/Ground Truth/control_points_{ids}_1_2.txt'
            lm = torch.from_numpy(np.genfromtxt(fname)) * (256 / 2912)
        points1, points2 = lm[:, :2], lm[:, 2:]
        return points1, points2

    def __getitem__(self, index):
        f, m = self.fixed[index], self.moving[index]
        f_v, m_v = self.fixed_vessels[index], self.moving_vessels[index]
        id = f.split('/')[-1].split('_')[0]

        # load images
        fixed_image = self.load_image(f)
        moving_image = self.load_image(m)

        # load segs
        fixed_vessel = self.load_image(f_v)
        moving_vessel = self.load_image(m_v)

        # load keypoints
        fixed_kp, moving_kp = self.load_kp(id, data=self.landmarks)

        return fixed_image, moving_image, fixed_vessel, moving_vessel, fixed_kp, moving_kp
    
    def __len__(self):
        return len(self.data)

class LambdaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2)
        self.linear = nn.Linear(in_features=2048*2 - 1, out_features=1)
        
    def forward(self, input1, input2):
        b = input1.shape[0]
        # stack inputs
        x1 = input1 # (b, n, 2)
        x2 = input2 # (b, n, 2)
        x = torch.stack([x1, x2], dim=1) # (b, 2, n, 2)
        x = torch.nn.functional.pad(x, pad=(0, 0, 2048-x.shape[2], 0))
        x = x.view(b, 2, 2048*2)
        x = self.conv1(x)
        x = self.linear(x.view(b, -1))
        return x

class TwoImageCNN(nn.Module):
    def __init__(self):
        super(TwoImageCNN, self).__init__()
        
        # Define convolutional layers for the first image
        self.conv1_img1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2_img1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool_img1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define convolutional layers for the second image
        self.conv1_img2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2_img2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool_img2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(32 * 64 * 64 * 2, 128)  # Adjust the input size based on the final feature map size
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x1, x2):
        # Forward pass for the first image
        x1 = torch.relu(self.conv1_img1(x1))
        x1 = self.pool_img1(x1)
        x1 = torch.relu(self.conv2_img1(x1))
        x1 = self.pool_img1(x1)
        
        # Forward pass for the second image
        x2 = torch.relu(self.conv1_img2(x2))
        x2 = self.pool_img2(x2)
        x2 = torch.relu(self.conv2_img2(x2))
        x2 = self.pool_img2(x2)
        
        # Concatenate the feature maps from the two images
        x = torch.cat((x1, x2), dim=1)
        
        # Flatten the feature maps
        x = x.view(-1, 32 * 64 * 64 * 2)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def main(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # =========================
    # make experiment directory
    # =========================
    os.makedirs(args.save_dir, exist_ok=True)

    # =====================================
    # 1. Load Dataset and Create Dataloader
    # =====================================
    dataset = ImageDataset(
        path=args.data, 
        fixed_img_col=args.fixed, 
        moving_img_col=args.moving,
        fixed_vessel_col=args.fixed_vessel,
        moving_vessel_col=args.moving_vessel,
        lm=args.landmarks
    )

    # convert to dataloader
    dataloader = DataLoader(dataset, shuffle=True, batch_size=16, num_workers=4)
    original_rnd_state = torch.random.get_rng_state()

    # ============================
    # 2. Load model and optimizers
    # ============================
    # model = TwoImageCNN().to(device)
    model = LambdaMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ==================
    # 3. Begin Training
    # ==================
    epoch_losses = []
    for e in range(50):
        print(f'Epoch {e+1}/{50}')

        batch_losses = []
        for batch_data in tqdm(dataloader):

            # get images
            fixed = batch_data[0]
            moving = batch_data[1]
            fixed_vessel = batch_data[2].to(device)
            moving_vessel = batch_data[3].to(device)
            fixed_kp = batch_data[4].to(device)
            moving_kp = batch_data[5].to(device)

            # # save keypoint matches
            # os.makedirs(os.path.join(args.save_dir, 'keypoint_matches'), exist_ok=True)
            # kp_match_save_path = os.path.join(args.save_dir, 'keypoint_matches', f'image_{i}.png')
            # # visualize keypoint correspondences
            # axes = viz2d.plot_images([fixed_vessel.cpu().squeeze(0), moving_vessel.cpu().squeeze(0)])
            # viz2d.plot_matches(fixed_kp.cpu().squeeze(0), moving_kp.cpu().squeeze(0), color="lime", lw=0.2)
            # plt.savefig(kp_match_save_path)

            # ===============================
            # 3.1: Predict lambda from images
            # ===============================
            # lambdas = model(fixed_vessel, moving_vessel) # (B, 1)

            # ========================
            # 3.2: Registration module
            # ========================
            fixed_kp = normalize_coordinates(fixed_kp, fixed.shape[2:]).float()
            moving_kp = normalize_coordinates(moving_kp, moving.shape[2:]).float()
            lambdas = model(fixed_kp, moving_kp)
            
            # compute registration
            theta, grid = compute_tps(moving_kp, fixed_kp, fixed.shape, lambdas) # image register
            keypoints_registered_manual = TPS(dim=2).points_from_points(
                    moving_kp, 
                    fixed_kp, 
                    moving_kp, 
                    lmbda=lambdas
                ) # keypoint register
            keypoints_registered_manual = unnormalize_coordinates(keypoints_registered_manual, fixed.shape[2:])
            fixed_kp = unnormalize_coordinates(fixed_kp, fixed.shape[2:])
            moving_kp = unnormalize_coordinates(moving_kp, moving.shape[2:])

            mask = align_img(grid, torch.ones_like(fixed).to(device))
            registered = align_img(grid, moving_vessel)

            # ==============================
            # 3.3: Compute loss and backprop
            # ==============================
            if args.loss == 'dice':
                loss = DiceLoss(squared_pred=True)(registered * mask, fixed_vessel * mask)
            else:
                loss = nn.MSELoss()(keypoints_registered_manual, fixed_kp)
            batch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ===================
        # 3.4: Visualize loss
        # ===================
        mean_batch_loss = torch.mean(torch.tensor(batch_losses)).item()
        epoch_losses.append(mean_batch_loss)

        plt.figure(figsize=(12, 6))
        plt.plot(epoch_losses)
        plt.savefig(os.path.join(args.save_dir, 'performance.png'))
        plt.close()

        # save to json
        with open(os.path.join(args.save_dir, 'performance.json'), 'w+') as f:
            json.dump(epoch_losses, f)

        # ======================
        # 3.5: Visualize results
        # ======================
        os.makedirs(os.path.join(args.save_dir, 'checkerboards'), exist_ok=True)
        for i in range(len(registered)):
            ckbd = create_checkerboard(fixed_vessel[i], registered[i])
            ckbd = ToPILImage()(ckbd)
            plt.figure()
            plt.imshow(ckbd)
            plt.title(f'Predicted Lambda: {lambdas[i].item():.2f}, {args.loss} loss: {batch_losses[i]:.2f}')
            plt.savefig(os.path.join(args.save_dir, 'checkerboards', f'sample_{i}_epoch_{e}'))
            plt.close()

        # =============
        # Save weights
        # =============
        torch.save(model, os.path.join(args.save_dir, 'weights.pth'))

        # =========================
        # Reset state of dataloader
        # =========================
        torch.random.set_rng_state(original_rnd_state)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('-d', '--data', type=str, help='Dataset csv path')
    parser.add_argument('-f', '--fixed', default='fixed_image', type=str, help='Fixed image column')
    parser.add_argument('-m', '--moving', default='moving_image', type=str, help='Moving image column')
    parser.add_argument('-fv', '--fixed-vessel', default=None, type=none_or_str, help='Fixed vessel column')
    parser.add_argument('-mv', '--moving-vessel', default=None, type=none_or_str, help='Moving vessel column')    
    parser.add_argument('-l', '--landmarks', help='Ground Truth Landmarks source', default=None, type=str)
    parser.add_argument('--loss', help='Loss function', default='mse', type=str)
    parser.add_argument('--seed', type=int, help='Seed for reproducibility.', default=1399)
    
    # others
    parser.add_argument('--save-dir', type=str, default='results_uchealth/', help='Save location for images and csv')
    parser.add_argument('--device', default='cpu', help='Device to run program on')
    args = parser.parse_args()
    
    main(args)
    print(f'Results Saved to: {args.save_dir}')