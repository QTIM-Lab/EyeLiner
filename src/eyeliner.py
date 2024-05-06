''' Register two images using the EyeLiner API '''

# install libraries
import cv2
import torch
from torch.nn import functional as F
from .utils import normalize_coordinates, unnormalize_coordinates, TPS
from .detectors import get_keypoints_splg

class EyeLinerP():

    ''' API for pairwise retinal image registration '''

    def __init__(self, reg='affine', lambda_tps=1., image_size=(3, 256, 256), device='cpu'):
        self.reg = reg
        self.lambda_tps = lambda_tps
        self.image_size = image_size
        self.device = device
    
    def get_corr_keypoints(self, fixed_image, moving_image):
        try:
            keypoints_fixed, keypoints_moving = get_keypoints_splg(fixed_image, moving_image)
            n = keypoints_fixed.shape[1]
            if n < 3:
                print(f'Found {n} keypoints only! Cannot register!')
        except:
            print('Not able to register image pair!')

        return keypoints_fixed, keypoints_moving
    
    def compute_lq_affine(self, points0, points1):
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

        return A.cpu()

    def compute_tps(self, keypoints_moving, keypoints_fixed, grid_shape, lmbda):
        theta, grid = TPS(dim=2).grid_from_points(keypoints_moving, keypoints_fixed, grid_shape=grid_shape, lmbda=torch.tensor(lmbda).to(self.device))   
        return theta.cpu(), grid.cpu()
    
    def get_registration(self, kp_fixed, kp_moving):

        if self.reg == 'affine':
            # compute least squares solution using key points
            theta = self.compute_lq_affine(kp_fixed, kp_moving).float()

        elif self.reg == 'tps':
            # scale between -1 and 1
            keypoints_fixed = normalize_coordinates(kp_fixed, self.image_size[1:])
            keypoints_moving = normalize_coordinates(kp_moving, self.image_size[1:])
            theta = self.compute_tps(keypoints_moving, keypoints_fixed, [1] + list(self.image_size), self.lambda_tps)
            keypoints_fixed = unnormalize_coordinates(keypoints_fixed, self.image_size[1:])
            keypoints_moving = unnormalize_coordinates(keypoints_moving, self.image_size[1:])

        else:
            raise NotImplementedError('Only affine and thin-plate spline registration supported.')
    
        return theta

    def apply_transform(self, theta, moving_image):

        if self.reg == 'affine':
            warped_image = torch.permute(moving_image, (1, 2, 0)).numpy() # (h, w, c)
            affine_mat = theta.numpy() # (3, 3)
            warped_image = cv2.warpAffine(warped_image, affine_mat[:2, :], (warped_image.shape[0], warped_image.shape[1]))

        elif self.reg == 'tps':
            warped_image = F.grid_sample(
                moving_image, grid=theta, mode="bilinear", padding_mode="zeros", align_corners=False
            )

        else:
            raise NotImplementedError('Only affine and thin-plate spline registration supported.')

        return warped_image

    def __call__(self, data):

        # extract data
        fixed_image = data['fixed_image'].to(self.device)
        moving_image = data['moving_image'].to(self.device)

        # 1. Blood vessel and optic disk extraction
        # fixed_vessels = self.vessel_net(fixed_image)
        # moving_vessels = self.vessel_net(moving_image)
        # fixed_disk = self.disk_net(fixed_image)
        # moving_disk = self.disk_net(moving_image)

        # 2. Deep Keypoint Detection
        kp_fixed, kp_moving = self.get_corr_keypoints(fixed_image, moving_image)

        # 3. Registration module
        theta = self.get_registration(kp_fixed, kp_moving)

        return theta

class EyeLinerS():

    ''' API for retinal image registration '''