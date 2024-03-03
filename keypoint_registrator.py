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
from scipy.stats import zscore

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, Grayscale
from torch.nn import functional as F

from src.data import ImageDataset
from src.visualize import create_checkerboard, visualize_deformation_grid

from src.detectors import get_keypoints
from lightglue import viz2d
from matplotlib import pyplot as plt

from monai.losses import BendingEnergyLoss
from src.lambda_cnn import LambdaMLP

# =================
# Utility functions
# =================

class TPS:       
  '''See https://github.com/cheind/py-thin-plate-spline/blob/master/thinplate/numpy.py'''
  def __init__(self, dim):
      self.dim = dim

  def fit(self, c, lmbda):        
      '''Assumes last dimension of c contains target points.
      
        Set up and solve linear system:
          [K   P] [w] = [v]
          [P^T 0] [a]   [0]
      Args:
        c: control points and target point (bs, T, d+1)
        lmbda: Lambda values per batch (bs)
      '''
      device = c.device
      bs, T = c.shape[0], c.shape[1]
      ctrl, tgt = c[:, :, :self.dim], c[:, :, -1]

      # Build K matrix
      U = TPS.u(TPS.d(ctrl, ctrl))
      I = torch.eye(T).repeat(bs, 1, 1).float().to(device)
      K = U + I*lmbda.view(bs, 1, 1)

      # Build P matrix
      P = torch.ones((bs, T, self.dim+1)).float()
      P[:, :, 1:] = ctrl

      # Build v vector
      v = torch.zeros(bs, T+self.dim+1).float()
      v[:, :T] = tgt

      A = torch.zeros((bs, T+self.dim+1, T+self.dim+1)).float()
      A[:, :T, :T] = K
      A[:, :T, -(self.dim+1):] = P
      A[:, -(self.dim+1):, :T] = P.transpose(1,2)

      theta = torch.linalg.solve(A, v) # p has structure w,a
      return theta
  
  @staticmethod
  def d(a, b):
      '''Compute pair-wise distances between points.
      
      Args:
        a: (bs, num_points, d)
        b: (bs, num_points, d)
      Returns:
        dist: (bs, num_points, num_points)
      '''
      return torch.sqrt(torch.square(a[:, :, None, :] - b[:, None, :, :]).sum(-1) + 1e-6)

  @staticmethod
  def u(r):
      '''Compute radial basis function.'''
      return r**2 * torch.log(r + 1e-6)
  
  def tps_theta_from_points(self, c_src, c_dst, lmbda):
      '''
      Args:
        c_src: (bs, T, dim)
        c_dst: (bs, T, dim)
        lmbda: (bs)
      '''
      device = c_src.device
      
      cx = torch.cat((c_src, c_dst[..., 0:1]), dim=-1)
      cy = torch.cat((c_src, c_dst[..., 1:2]), dim=-1)
      if self.dim == 3:
          cz = torch.cat((c_src, c_dst[..., 2:3]), dim=-1)

      theta_dx = self.fit(cx, lmbda).to(device)
      theta_dy = self.fit(cy, lmbda).to(device)
      if self.dim == 3:
          theta_dz = self.fit(cz, lmbda).to(device)

      if self.dim == 3:
          return torch.stack((theta_dx, theta_dy, theta_dz), -1)
      else:
          return torch.stack((theta_dx, theta_dy), -1)

  def tps(self, theta, ctrl, grid):
      '''Evaluate the thin-plate-spline (TPS) surface at xy locations arranged in a grid.
      The TPS surface is a minimum bend interpolation surface defined by a set of control points.
      The function value for a x,y location is given by
      
        TPS(x,y) := theta[-3] + theta[-2]*x + theta[-1]*y + \sum_t=0,T theta[t] U(x,y,ctrl[t])
        
      This method computes the TPS value for multiple batches over multiple grid locations for 2 
      surfaces in one go.
      
      Params
      ------
      theta: Nx(T+3)xd tensor, or Nx(T+2)xd tensor
        Batch size N, T+3 model parameters for T control points in dx and dy.
      ctrl: NxTxd tensor
        T control points in normalized image coordinates [0..1]
      grid: NxHxWx(d+1) tensor
        Grid locations to evaluate with homogeneous 1 in first coordinate.
        
      Returns
      -------
      z: NxHxWxd tensor
        Function values at each grid location in dx and dy.
      '''
      
      if len(grid.shape) == 4:
          N, H, W, _ = grid.size()
          diff = grid[...,1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1)
      else:
          N, D, H, W, _ = grid.size()
          diff = grid[...,1:].unsqueeze(-2) - ctrl.unsqueeze(1).unsqueeze(1).unsqueeze(1)

      T = ctrl.shape[1]
      
      pair_dist = torch.sqrt((diff**2).sum(-1))
      U = TPS.u(pair_dist)

      w, a = theta[:, :-(self.dim+1), :], theta[:, -(self.dim+1):, :]

      # U is NxHxWxT
      # b contains dot product of each kernel weight and U(r)
      b = torch.bmm(U.view(N, -1, T), w)
      if len(grid.shape) == 4:
          b = b.view(N,H,W,self.dim)
      else:
          b = b.view(N,D,H,W,self.dim)
      
      # b is NxHxWxd
      # z contains dot product of each affine term and polynomial terms.
      z = torch.bmm(grid.view(N,-1,self.dim+1), a)
      if len(grid.shape) == 4:
          z = z.view(N,H,W,self.dim) + b
      else:
          z = z.view(N,D,H,W,self.dim) + b
      return z

  def tps_grid(self, theta, ctrl, size):
      '''Compute a thin-plate-spline grid from parameters for sampling.
      
      Params
      ------
      theta: Nx(T+3)x2 tensor
        Batch size N, T+3 model parameters for T control points in dx and dy.
      ctrl: NxTx2 tensor, or Tx2 tensor
        T control points in normalized image coordinates [0..1]
      size: tuple
        Output grid size as NxCxHxW. C unused. This defines the output image
        size when sampling.
      
      Returns
      -------
      grid : NxHxWx2 tensor
        Grid suitable for sampling in pytorch containing source image
        locations for each output pixel.
      '''    
      device = theta.device
      if len(size) == 4:
          N, _, H, W = size
          grid_shape = (N, H, W, self.dim+1)
      else:
          N, _, D, H, W = size
          grid_shape = (N, D, H, W, self.dim+1)
      grid = self.uniform_grid(grid_shape).to(device)
      
      z = self.tps(theta, ctrl, grid)
      return z 

  def uniform_grid(self, shape):
      '''Uniform grid coordinates.
      
      Params
      ------
      shape : tuple
          NxHxWx3 defining the batch size, height and width dimension of the grid.
          3 is for the number of dimensions (2) plus 1 for the homogeneous coordinate.
      Returns
      -------
      grid: HxWx3 tensor
          Grid coordinates over [-1,1] normalized image range.
          Homogenous coordinate in first coordinate position.
          After that, the second coordinate varies first, then
          the third coordinate varies, then (optionally) the 
          fourth coordinate varies.
      '''

      if self.dim == 2:
          _, H, W, _ = shape
      else:
          _, D, H, W, _ = shape
      grid = torch.zeros(shape)

      grid[..., 0] = 1.
      grid[..., 1] = torch.linspace(-1, 1, W)
      grid[..., 2] = torch.linspace(-1, 1, H).unsqueeze(-1)   
      if grid.shape[-1] == 4:
          grid[..., 3] = torch.linspace(-1, 1, D).unsqueeze(-1).unsqueeze(-1)  
      return grid
  
  def grid_from_points(self, ctl_points, tgt_points, grid_shape, **kwargs):
      lmbda = kwargs['lmbda']
      theta = self.tps_theta_from_points(tgt_points, ctl_points, lmbda)
      grid = self.tps_grid(theta, tgt_points, grid_shape)
      return theta, grid

  def deform_points(self, theta, ctrl, points):
      weights, affine = theta[:, :-(self.dim+1), :], theta[:, -(self.dim+1):, :]
      N, T, _ = ctrl.shape
      U = TPS.u(TPS.d(ctrl, points))

      P = torch.ones((N, points.shape[1], self.dim+1)).float().to(theta.device)
      P[:, :, 1:] = points[:, :, :self.dim]

      # U is NxHxWxT
      b = torch.bmm(U.transpose(1, 2), weights)
      z = torch.bmm(P.view(N,-1,self.dim+1), affine)
      return z + b
  
  def points_from_points(self, ctl_points, tgt_points, points, **kwargs):
      lmbda = kwargs['lmbda']
      theta = self.tps_theta_from_points(ctl_points, tgt_points, lmbda)
      return self.deform_points(theta, ctl_points, points)
  
def normalize_coordinates(coords, shape):
    coords_ = coords.clone()
    num_rows, num_cols = shape
    coords_[:, :, 1] /= (num_cols - 1.) # scales between 0-1
    coords_[:, :, 0] /= (num_rows - 1.) # scales between 0-1
    coords_ = 2 * coords_ - 1 
    return coords_

def unnormalize_coordinates(coords, shape):
    coords_ = coords.clone()
    num_rows, num_cols = shape
    coords_ = (coords_ + 1) / 2
    coords_[:, :, 1] *= (num_cols - 1.)
    coords_[:, :, 0] *= (num_rows - 1.)
    return coords_

def compute_auc_single(error):
    error = np.array(error)
    limit = 25
    gs_error = np.zeros(limit + 1)

    accum = 0
    for i in range(1, limit + 1):
        gs_error[i] = np.sum(error < i) * 100 / len(error)
        accum = accum + gs_error[i]
    auc = accum / (limit * 100)
    return auc

# Compute AUC scores for image registration on the FIRE dataset
def compute_auc(s_error, p_error, a_error):
    # assert (len(s_error) == 71)  # Easy pairs
    # assert (len(p_error) == 48)  # Hard pairs. Note file control_points_P37_1_2.txt is ignored
    # assert (len(a_error) == 14)  # Moderate pairs

    s_error = np.array(s_error)
    p_error = np.array(p_error)
    a_error = np.array(a_error)

    limit = 25
    gs_error = np.zeros(limit + 1)
    gp_error = np.zeros(limit + 1)
    ga_error = np.zeros(limit + 1)

    accum_s = 0
    accum_p = 0
    accum_a = 0

    for i in range(1, limit + 1):
        gs_error[i] = np.sum(s_error < i) * 100 / len(s_error)
        gp_error[i] = np.sum(p_error < i) * 100 / len(p_error)
        ga_error[i] = np.sum(a_error < i) * 100 / len(a_error)

        accum_s = accum_s + gs_error[i]
        accum_p = accum_p + gp_error[i]
        accum_a = accum_a + ga_error[i]

    auc_s = accum_s / (limit * 100)
    auc_p = accum_p / (limit * 100)
    auc_a = accum_a / (limit * 100)
    mAUC = (auc_s + auc_p + auc_a) / 3.0
    return {'s': auc_s, 'p': auc_p, 'a': auc_a, 'mAUC': mAUC}

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
        fname = [f'data/retina_datasets/SIGF_time_variant_sequences/landmarks/{i}.csv' for i in ids]
        lm = [torch.tensor(pd.read_csv(f, header=None).values) for f in fname]
    elif data == 'uchealth':
        fname = [f'data/retina_datasets/UCHealth_Annotations/landmarks/{i}.csv' for i in ids]
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

def compute_tps(keypoints_moving, keypoints_fixed, grid_shape, lmbda):
    theta, grid = TPS(dim=2).grid_from_points(keypoints_moving, keypoints_fixed, grid_shape=grid_shape, lmbda=lmbda)   
    return theta, grid

def align_img(grid, x):
    return F.grid_sample(
        x, grid=grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

# outlier filtering
def remove_outliers(data, thresh=None):
    if thresh is not None:
        return data <= thresh
    q1 = np.percentile(data, 40, axis=1)
    q3 = np.percentile(data, 60, axis=1)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

def remove_outliers_matches(slopes, lengths, thresh=None):

    slope_outliers = remove_outliers(slopes, thresh)
    length_outliers = remove_outliers(lengths, thresh)

    # Combine outlier masks
    combined_outliers = slope_outliers | length_outliers

    return combined_outliers

def confidence_filter(kp1, kp2, shape):
    _, _, _, w = shape

    kp2_shifted = kp2.clone() 
    kp2_shifted[:, :, 0] += w

    # compute slope and length of line
    difference = kp2_shifted - kp1 # (b, n, 2)
    slopes = difference[:, :, 1] / (difference[:, :, 0] + 1e-15) # (b, n)
    # 176, 322
    lengths = torch.linalg.norm(difference, ord=2, dim=2) # (b, n)

    # remove outlier
    outliers = remove_outliers_matches(slopes.numpy(), lengths.numpy())

    # filter keypoints
    filtered_kp1 = kp1[:, ~outliers.squeeze()]
    filtered_kp2 = kp2[:, ~outliers.squeeze()]

    return filtered_kp1, filtered_kp2

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
    os.makedirs(os.path.join(args.save, 'histograms'), exist_ok=True)
    os.makedirs(os.path.join(args.save, 'quiver_plots'), exist_ok=True)
    
    results_df = pd.DataFrame(columns=['Fixed', 'Registered', 'Fixed_Vessels', 'Registered_Vessels', 'Fixed_Disks', 'Registered_Disks', 'Difference Map'])
    fixed_images = []
    reg_images = []
    fixed_vessels_images = []
    reg_vessels_images = []
    fixed_disks_images = []
    reg_disks_images = []
    checkerboard_images = []
    if args.evaluate:
        registration_error_fixed_moving = []
        registration_error_fixed_reg = []
        if args.landmarks == 'fire':
            auc_record = dict([(category, []) for category in ['S', 'P', 'A']])
        else:
            auc_record = []
    diff_maps = []
    quiver_plots = []

    # compute the keypoint locations
    step = 0
    lambda_values = []
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

        # ==========================
        # 1. Deep Keypoint Detection
        # ==========================
        if args.manual:
            if args.landmarks == 'fire':
                ids = [f.split('/')[-1].split('_')[0] for f in fixed_paths]
            else:
                ids = [f.split('/')[-2] for f in fixed_paths]
            keypoints_fixed, keypoints_moving = get_steve_kp(ids, args.landmarks) # (b, n, 2)
            keypoints_fixed, keypoints_moving = keypoints_fixed.float(), keypoints_moving.float()
            keypoints_fixed_unfiltered, keypoints_moving_unfiltered = keypoints_fixed, keypoints_moving
            timing_detection = [0]
        else:
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
                        mask=args.mask,
                        top=args.top
                    )
                n = keypoints_fixed.shape[1]
                if n < 3:
                    print(f'Found {n} keypoints in {fixed_paths[0]} and {moving_paths[0]}! Cannot register!')
                    step += 1
                    continue
            except:
                print('Not able to register image pair!')
                continue

            # filter keypoints
            # keypoints_fixed, keypoints_moving = confidence_filter(keypoints_fixed, keypoints_moving, shape=fixed.shape)

        # =======================================================
        # 2. TODO: Iterative point refinement/Hyperparam Learning
        # =======================================================

        iterative_point_refinement = False 
        if iterative_point_refinement:
            # compute registered keypoints
            keypoints_moving_ = torch.cat([keypoints_moving, torch.ones(1, keypoints_moving.shape[1], 1)], dim=2)
            keypoints_moving_ = torch.permute(keypoints_moving_, (0, 2, 1)).float()
            keypoints_registered = keypoints_registered[:, :2, :] / keypoints_registered[:, 2, :][:, None, :] # (b, 2, n)
            keypoints_registered = torch.permute(keypoints_registered, (0, 2, 1)) # (b, n, 2)

            # compute distances
            rmse_fr = torch.sqrt(torch.sum((keypoints_fixed - keypoints_registered)**2, dim=-1)).ravel()
            min_, max_ = min(rmse_fr), max(rmse_fr)

            # get rid of outlier points
            inliers = remove_outliers(rmse_fr.numpy(), 2.0)
            keypoints_fixed =  keypoints_fixed[0][inliers].unsqueeze(0)
            keypoints_moving = keypoints_moving[0][inliers].unsqueeze(0)
            A = compute_lq_affine(keypoints_fixed, keypoints_moving).float()

            # compute registered keypoints
            keypoints_moving_ = torch.cat([keypoints_moving, torch.ones(1, keypoints_moving.shape[1], 1)], dim=2)
            keypoints_moving_ = torch.permute(keypoints_moving_, (0, 2, 1)).float()
            # keypoints_registered = torch.bmm(A[:, :2, :], keypoints_moving_) # (b, 2, n)
            keypoints_registered = torch.bmm(A, keypoints_moving_) # (b, 3, n)
            keypoints_registered = keypoints_registered[:, :2, :] / keypoints_registered[:, 2, :][:, None, :] # (b, 2, n)
            keypoints_registered = torch.permute(keypoints_registered, (0, 2, 1)) # (b, n, 2)
            
            rmse_filtered = rmse_fr.ravel()[inliers]
            rmse_fr = torch.sqrt(torch.sum((keypoints_fixed - keypoints_registered)**2, dim=-1)).ravel()

            # threshold
            plt.figure()
            plt.hist(rmse_filtered, bins=np.arange(min(rmse_filtered), max(rmse_filtered) + 0.1, 0.1))
            plt.xlim(0, max_+0.1)
            plt.savefig(os.path.join(args.save, f'histograms/hist_after_{step}_filtered.png'))
            plt.close()

            # threshold
            plt.figure()
            plt.hist(rmse_fr.ravel(), bins=np.arange(min(rmse_fr), max(rmse_fr) + 0.1, 0.1))
            plt.xlim(0, max_+0.1)
            plt.savefig(os.path.join(args.save, f'histograms/hist_after_{step}.png'))
            plt.close()

        # =======================
        # 3. Registration module
        # =======================

        # Start the timer for this iteration
        start_time = time.time()
        print(keypoints_fixed.shape, keypoints_moving.shape)

        if args.reg_method == 'affine':
            # # compute initial least squares affine using key points
            # A = compute_lq_affine(keypoints_fixed, keypoints_moving).float()

            # # compute registered keypoints
            # keypoints_moving_ = torch.cat([keypoints_moving, torch.ones(1, keypoints_moving.shape[1], 1)], dim=2)
            # keypoints_moving_ = torch.permute(keypoints_moving_, (0, 2, 1)).float()
            # # keypoints_registered = torch.bmm(A[:, :2, :], keypoints_moving_) # (b, 2, n)
            # keypoints_registered = torch.bmm(A, keypoints_moving_) # (b, 3, n)
            # keypoints_registered = keypoints_registered[:, :2, :] / keypoints_registered[:, 2, :][:, None, :] # (b, 2, n)
            # keypoints_registered = torch.permute(keypoints_registered, (0, 2, 1)) # (b, n, 2)

            # # compute distances
            # rmse_fr = torch.sqrt(torch.sum((keypoints_fixed - keypoints_registered)**2, dim=-1)).ravel()
            # min_, max_ = min(rmse_fr), max(rmse_fr)

            # # threshold
            # plt.figure()
            # plt.hist(rmse_fr.ravel(), bins=np.arange(min_, max_ + 0.1, 0.1))
            # plt.xlim(0, max_+0.1)
            # plt.savefig(os.path.join(args.save, f'histograms/hist_before_{step}.png'))
            # plt.close()

            # # get rid of outlier points
            # inliers = remove_outliers(rmse_fr.numpy(), 2.0)
            # keypoints_fixed =  keypoints_fixed[0][inliers].unsqueeze(0)
            # keypoints_moving = keypoints_moving[0][inliers].unsqueeze(0)
            # A = compute_lq_affine(keypoints_fixed, keypoints_moving).float()

            # # compute registered keypoints
            # keypoints_moving_ = torch.cat([keypoints_moving, torch.ones(1, keypoints_moving.shape[1], 1)], dim=2)
            # keypoints_moving_ = torch.permute(keypoints_moving_, (0, 2, 1)).float()
            # # keypoints_registered = torch.bmm(A[:, :2, :], keypoints_moving_) # (b, 2, n)
            # keypoints_registered = torch.bmm(A, keypoints_moving_) # (b, 3, n)
            # keypoints_registered = keypoints_registered[:, :2, :] / keypoints_registered[:, 2, :][:, None, :] # (b, 2, n)
            # keypoints_registered = torch.permute(keypoints_registered, (0, 2, 1)) # (b, n, 2)
            
            # rmse_filtered = rmse_fr.ravel()[inliers]
            # rmse_fr = torch.sqrt(torch.sum((keypoints_fixed - keypoints_registered)**2, dim=-1)).ravel()

            # # threshold
            # plt.figure()
            # plt.hist(rmse_filtered, bins=np.arange(min(rmse_filtered), max(rmse_filtered) + 0.1, 0.1))
            # plt.xlim(0, max_+0.1)
            # plt.savefig(os.path.join(args.save, f'histograms/hist_after_{step}_filtered.png'))
            # plt.close()

            # # threshold
            # plt.figure()
            # plt.hist(rmse_fr.ravel(), bins=np.arange(min(rmse_fr), max(rmse_fr) + 0.1, 0.1))
            # plt.xlim(0, max_+0.1)
            # plt.savefig(os.path.join(args.save, f'histograms/hist_after_{step}.png'))
            # plt.close()

            # ==========================
            # One step affine prediction
            # ==========================

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

            # ==========================
        elif args.reg_method == 'tps':
            # scale between -1 and 1
            keypoints_fixed = normalize_coordinates(keypoints_fixed, fixed.shape[2:])
            keypoints_moving = normalize_coordinates(keypoints_moving, moving.shape[2:])
            if args.lambda_tps is not None:
                lambdas = torch.tensor(args.lambda_tps)
            else:
                # load lambda predictor
                model = torch.load('trained_models_new/lambda_predictor_4/weights.pth').to('cpu')
                with torch.no_grad():
                    lambdas = model(keypoints_fixed, keypoints_moving)
            lambda_values.append(lambdas)
            theta, grid = compute_tps(keypoints_moving, keypoints_fixed, fixed.shape, lambdas)
            keypoints_fixed = unnormalize_coordinates(keypoints_fixed, fixed.shape[2:])
            keypoints_moving = unnormalize_coordinates(keypoints_moving, moving.shape[2:])
        else:
            raise NotImplementedError

        end_time = time.time()
        timing_registration = end_time - start_time
        runtime = sum(timing_detection + [timing_registration])
        timings.append(runtime)

        # =============================
        # Evaluation and Saving Results
        # =============================

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

        print(f'Runtime per image: {runtime}s')

        # use Steve's landmarks for evaluation (if not already using his)
        if args.evaluate:
            if args.landmarks == 'fire':
                ids = [f.split('/')[-1].split('_')[0] for f in fixed_paths]
            else:
                ids = [f.split('/')[-2] for f in fixed_paths]
            keypoints_fixed_manual, keypoints_moving_manual = get_steve_kp(ids, args.landmarks) # (b, n, 2)
            keypoints_fixed_manual, keypoints_moving_manual = keypoints_fixed_manual.float(), keypoints_moving_manual.float()

            if args.reg_method == 'affine':
                # compute predictions
                keypoints_moving_ = torch.cat([keypoints_moving_manual, torch.ones(1, keypoints_moving_manual.shape[1], 1)], dim=2)
                keypoints_moving_ = torch.permute(keypoints_moving_, (0, 2, 1)).float()
                keypoints_registered_manual = torch.bmm(A[:, :2, :], keypoints_moving_) # (b, 2, n)
                # keypoints_registered = torch.bmm(A, keypoints_moving_) # (b, 3, n)
                # keypoints_registered = keypoints_registered[:, :2, :] / keypoints_registered[:, 2, :][:, None, :] # (b, 2, n)
                keypoints_registered_manual = torch.permute(keypoints_registered_manual, (0, 2, 1)) # (b, n, 2)
            elif args.reg_method == 'tps':
                keypoints_registered_manual = TPS(dim=2).points_from_points(
                    normalize_coordinates(keypoints_moving, moving.shape[2:]), 
                    normalize_coordinates(keypoints_fixed, fixed.shape[2:]), 
                    normalize_coordinates(keypoints_moving_manual, moving.shape[2:]), 
                    lmbda=lambdas
                )
                keypoints_registered_manual = unnormalize_coordinates(keypoints_registered_manual, fixed.shape[2:])
            else:
                raise NotImplementedError

        # post-process keypoints
        if args.evaluate:
            if args.landmarks == 'fire':
                keypoints_fixed_manual = keypoints_fixed_manual * (2912 / 256)
                keypoints_moving_manual = keypoints_moving_manual * (2912 / 256)
                keypoints_registered_manual = keypoints_registered_manual * (2912 / 256)

            # compute the error
            rmse_fm = torch.sqrt(torch.sum((keypoints_fixed_manual - keypoints_moving_manual)**2, dim=-1)).mean()
            rmse_fr = torch.sqrt(torch.sum((keypoints_fixed_manual - keypoints_registered_manual)**2, dim=-1)).mean()

            mae_fm = torch.mean(torch.abs(keypoints_fixed_manual - keypoints_moving_manual), dim=(-2, -1))
            mae_fr = torch.mean(torch.abs(keypoints_fixed_manual - keypoints_registered_manual), dim=(-2, -1))
            
            registration_error_fixed_moving.append(torch.mean(rmse_fm).item())
            registration_error_fixed_reg.append(torch.mean(rmse_fr).item())

            if args.landmarks == 'fire':
                category = [f.split('/')[-1][0] for f in fixed_paths]
                for c in category:
                    auc_record[c].append(rmse_fr)
            else:
                auc_record.append(rmse_fr)

        # warp moving images using affine matrix
        if args.reg_method == 'affine':
            mask = warp_affine(torch.ones_like(fixed), A)
            registered = warp_affine(moving, A)
            registered_corrected = warp_affine(moving_corrected, A)
            registered_vessels = warp_affine(moving_vessels, A)
            registered_disks = warp_affine(moving_disks, A)
        elif args.reg_method == 'tps':
            mask = align_img(grid, torch.ones_like(fixed))
            registered = align_img(grid, moving)
            registered_corrected = align_img(grid, moving_corrected)
            registered_vessels = align_img(grid, moving_vessels)
            registered_disks = align_img(grid, moving_disks)
        else:
            raise NotImplementedError

        registered = registered * mask
        registered_vessels = registered_vessels * mask
        registered_disks = registered_disks * mask
        fixed = fixed * mask
        moving = moving * mask

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
            quiver_maps_save_path = os.path.join(os.path.join(args.save, 'quiver_plots'), f'{step}_' + os.path.basename(moving_paths[i]).split('.')[0] + '.png')

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

            # save reg image (with landmarks annotated)
            if args.evaluate:
                if args.landmarks == 'fire':
                    lm = keypoints_registered_manual[i] * (2912 / 256)
                else:
                    lm = keypoints_registered_manual[i]
                reg = draw_coordinates(reg, lm, shape='+')
                reg.save(reg_w_lm_save_path)

            # save fixed image
            fixed = ToPILImage()(fixed[i])
            fixed.save(fixed_save_path)

            # save fixed image
            moving = ToPILImage()(moving[i])
            moving.save(moving_save_path)

            # save fixed image (with landmarks annotated)
            if args.evaluate:
                if args.landmarks == 'fire':
                    lm = keypoints_fixed_manual[i] * (2912 / 256)
                else:
                    lm = keypoints_fixed_manual[i]
                fixed = draw_coordinates(fixed, lm, shape='+')
                fixed.save(fixed_w_lm_save_path)

            # save moving image (with landmarks annotated)
            if args.evaluate:
                if args.landmarks == 'fire':
                    lm = keypoints_moving_manual[i] * (2912 / 256)
                else:
                    lm = keypoints_moving_manual[i]
                moving = draw_coordinates(moving, lm, shape='+')
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

            # save quiver plots for deformation
            if args.reg_method == 'tps':
                visualize_deformation_grid(grid[i], (256, 256), save_to=quiver_maps_save_path)

            fixed_images.append(fixed_save_path)
            reg_images.append(reg_save_path)
            checkerboard_images.append(cb_after_save_path)
            fixed_disks_images.append(fixed_disks_save_path)
            reg_disks_images.append(reg_disks_save_path)
            fixed_vessels_images.append(fixed_vessels_save_path)
            reg_vessels_images.append(reg_vessels_save_path)
            diff_maps.append(diff_maps_save_path)
            quiver_plots.append(quiver_maps_save_path)

            step += 1

    # save to df
    results_df['Fixed'] = fixed_images
    results_df['Registered'] = reg_images
    results_df['Fixed_Vessels'] = fixed_vessels_images
    results_df['Registered_Vessels'] = reg_vessels_images
    results_df['Fixed_Disks'] = fixed_disks_images
    results_df['Registered_Disks'] = reg_disks_images

    if args.evaluate:
        results_df['TRE_fixed_moving'] = registration_error_fixed_moving
        results_df['TRE_fixed_registered'] = registration_error_fixed_reg

        # compute mAUC
        if args.landmarks == 'fire':
            auc = compute_auc(auc_record['S'], auc_record['P'], auc_record['A'])
            print('S: %.3f, P: %.3f, A: %.3f, mAUC: %.3f' % (auc['s'], auc['p'], auc['a'], auc['mAUC']))
        else:
            auc = compute_auc_single(auc_record)
            print('mAUC: %.3f' % auc)

    results_df.to_csv(os.path.join(args.save, 'results.csv'), index=False)

    print(f'Average runtime per image: {mean(timings)}s')

    print(lambda_values)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('-d', '--data', default='UCHealth_Annotations/grant_images_pairs_wmasks__.csv', type=str, help='Dataset csv path')
    parser.add_argument('-f', '--fixed', default='fixed_image', type=str, help='Fixed column')
    parser.add_argument('-m', '--moving', default='moving_image', type=str, help='Moving column')
    parser.add_argument('-fv', '--fixed-vessel', default='fixed_mask', type=none_or_str, help='Fixed vessel column')
    parser.add_argument('-mv', '--moving-vessel', default='moving_mask', type=none_or_str, help='Moving vessel column')
    parser.add_argument('-fd', '--fixed-disk', default=None, type=none_or_str, help='Fixed disk column')
    parser.add_argument('-md', '--moving-disk', default=None, type=none_or_str, help='Moving disk column')
    
    # keypoint detector args
    parser.add_argument('--kp_method', help='Keypoint detection method', choices=['seg', 'superpoint', 'loftr'])
    parser.add_argument('--desc_method', help='Descriptor computation method', choices=['sift', 'superpoint', 'loftr'])
    parser.add_argument('--match_method', help='Descriptor matching method', choices=['lightglue_sift', 'lightglue_superpoint', 'bf', 'flann', 'loftr'])
    parser.add_argument('--input', help='Input image to keypoint detector', default='img', choices=['img', 'vmask', 'dmask', 'structural'])
    parser.add_argument('--mask', help='Mask out certain predited keypoints', default=None, choices=['vmask', 'dmask', 'structural'])
    parser.add_argument('--top', help='Select only top N confident keypoint matches', type=int, default=None)
    parser.add_argument('--reg_method', help='Registration method', type=str, default='affine')
    parser.add_argument('--lambda_tps', help='TPS lambda parameter', type=float, default=None)

    # visualize args
    parser.add_argument('-l', '--landmarks', help='Ground Truth Landmarks source', default=None, type=str)
    parser.add_argument('--manual', help='Use manually annotated landmarks for registration.', action='store_true')
    
    # others
    parser.add_argument('-e', '--evaluate', help='Flag for whether to compute landmark error or not.', action='store_true')
    parser.add_argument('-s', '--size', type=int, default=256, help='Size of images')
    parser.add_argument('--save', type=str, default='results_uchealth/', help='Save location for images and csv')
    parser.add_argument('--device', default='cpu', help='Device to run program on')
    args = parser.parse_args()
    
    main(args)
    print(f'Results Saved to: {args.save}')