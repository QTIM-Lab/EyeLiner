''' Register two images using the EyeLiner API '''

# install libraries
import cv2
import torch
from torch.nn import functional as F
from .utils import normalize_coordinates, unnormalize_coordinates, TPS
from .detectors import get_keypoints_splg, get_keypoints_loftr
from kornia.geometry.ransac import RANSAC

class EyeLinerP():

    ''' API for pairwise retinal image registration '''

    def __init__(self, kp_method='splg', reg='affine', lambda_tps=1., image_size=(3, 256, 256), device='cpu'):
        self.kp_method = kp_method
        self.reg = reg
        self.lambda_tps = lambda_tps
        self.image_size = image_size
        self.device = device
    
    def get_corr_keypoints(self, fixed_image, moving_image):
        if self.kp_method == 'splg':
            keypoints_fixed, keypoints_moving = get_keypoints_splg(fixed_image, moving_image)
        else:
            keypoints_fixed, keypoints_moving = get_keypoints_loftr(fixed_image, moving_image)
        return keypoints_fixed, keypoints_moving
    
    def compute_lq_affine(self, points0, points1):
        # convert to homogenous coordinates
        P = torch.cat([points0, torch.ones(1, points0.shape[1], 1).to(self.device)], dim=2) # (b, n, 3)
        P = torch.permute(P, (0, 2, 1)) # (b, 3, n)
        Q = torch.cat([points1, torch.ones(1, points1.shape[1], 1).to(self.device)], dim=2) # (b, n, 3)
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

    def KPRefiner(self, keypoints_fixed, keypoints_moving):
        _, mask = RANSAC(model_type='homography')(keypoints_fixed.squeeze(0), keypoints_moving.squeeze(0))
        mask = mask.squeeze()
        keypoints_fixed_filtered = keypoints_fixed[:, mask]
        keypoints_moving_filtered = keypoints_moving[:, mask]
        return keypoints_fixed_filtered, keypoints_moving_filtered

    @staticmethod
    def apply_transform(theta, moving_image):

        if theta.shape == (3, 3):
            warped_image = torch.permute(moving_image, (1, 2, 0)).numpy() # (h, w, c)
            affine_mat = theta.numpy() # (3, 3)
            warped_image = cv2.warpAffine(warped_image, affine_mat[:2, :], (warped_image.shape[0], warped_image.shape[1]))
            if warped_image.ndim == 2: # adding extra dim for grayscale warp
                warped_image = warped_image[:, :, None]
            warped_image = torch.tensor(warped_image).permute(2, 0, 1)

        elif theta.shape == (moving_image.shape[1], moving_image.shape[2], 2):
            warped_image = F.grid_sample(
                moving_image.unsqueeze(0), grid=theta.unsqueeze(0), mode="bilinear", padding_mode="zeros", align_corners=False
            ).squeeze(0)

        else:
            raise NotImplementedError('Only affine and deformation fields supported.')

        return warped_image

    @staticmethod
    def apply_transform_points(theta, moving_keypoints, ctrl_keypoints=None, tgt_keypoints=None, lmbda=None):
        if theta.shape == (3, 3):
            moving_keypoints = torch.cat([moving_keypoints, torch.ones(moving_keypoints.shape[0], 1)], dim=1).T # (3, N)
            warped_kp = torch.mm(theta[:2, :], moving_keypoints).T # (2, 3) @ (3, N) = (2, N) --> (N, 2)

        else:
            # method 1: recompute transforms?
            moving_keypoints = normalize_coordinates(moving_keypoints.unsqueeze(0).float(), shape=(256, 256))
            tgt_keypoints = normalize_coordinates(tgt_keypoints.unsqueeze(0).float(), shape=(256, 256))
            ctrl_keypoints = normalize_coordinates(ctrl_keypoints.unsqueeze(0).float(), shape=(256, 256))
            warped_kp = TPS(dim=2).points_from_points(
                ctrl_keypoints,
                tgt_keypoints,
                moving_keypoints,
                lmbda=torch.tensor(lmbda),
            )
            warped_kp = unnormalize_coordinates(warped_kp, shape=(256, 256)).squeeze(0)
            
            # method 2: apply inverse transform to the fixed points
            # moving_keypoints = normalize_coordinates(moving_keypoints.unsqueeze(0).float(), shape=(256, 256))
            # ctrl_keypoints = normalize_coordinates(ctrl_keypoints.unsqueeze(0).float(), shape=(256, 256))
            # warped_kp = TPS(dim=2).deform_points(
            #     theta.unsqueeze(0),
            #     ctrl_keypoints,
            #     moving_keypoints
            # )
            # warped_kp = unnormalize_coordinates(warped_kp, shape=(256, 256)).squeeze(0)

        return warped_kp
    
    def __call__(self, data):

        # 1. extract data
        fixed_image = data['fixed_input'].to(self.device)
        moving_image = data['moving_input'].to(self.device)

        # 2. Deep Keypoint Detection
        kp_fixed, kp_moving = self.get_corr_keypoints(fixed_image, moving_image)

        # 3. Keypoint Refinement
        kp_fixed, kp_moving = self.KPRefiner(kp_fixed, kp_moving)

        # 4. Registration module
        theta = self.get_registration(kp_fixed, kp_moving)

        cache = {
            'kp_fixed': kp_fixed,
            'kp_moving': kp_moving
        }

        return theta, cache