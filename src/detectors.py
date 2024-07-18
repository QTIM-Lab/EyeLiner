import sys, time
import numpy as np
import torch
import cv2
import kornia as K
import kornia.feature as KF
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from PIL import Image
from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import load_image, rbd

def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel

def skeletonize_mask_disk(fixed_vessel, moving_vessel, fixed_disk, moving_disk):
    # skeletonize vessel
    fixed_vessel_skeleton = skeletonize(fixed_vessel)
    moving_vessel_skeleton = skeletonize(moving_vessel)
    
    # skeletonize disk
    fixed_disk_skeleton = cv2.distanceTransform(np.uint8(fixed_disk), cv2.DIST_C, 5)
    fixed_disk_skeleton = (fixed_disk_skeleton == 1).astype(np.uint8)
    moving_disk_skeleton = cv2.distanceTransform(np.uint8(moving_disk), cv2.DIST_C, 5)
    moving_disk_skeleton = (moving_disk_skeleton == 1).astype(np.uint8)
    return fixed_vessel_skeleton, moving_vessel_skeleton, fixed_disk_skeleton, moving_disk_skeleton

def get_intersection(fixed_disk, fixed_vessel, moving_disk, moving_vessel, thresh=128):
    
    # threshold masks
    fixed_vessel = np.uint8(fixed_vessel > thresh)
    fixed_disk = np.uint8(fixed_disk > thresh)
    moving_vessel = np.uint8(moving_vessel > thresh)
    moving_disk = np.uint8(moving_disk > thresh)
    
    # skeletonize masks
    fixed_vessel, moving_vessel, fixed_disk, moving_disk = skeletonize_mask_disk(
        fixed_vessel, 
        moving_vessel, 
        fixed_disk, 
        moving_disk
        )
    
    # compute intersection
    fixed_keypoints = np.argwhere(fixed_disk + fixed_vessel == 2)
    moving_keypoints = np.argwhere(moving_disk + moving_vessel == 2)
    return fixed_keypoints, moving_keypoints

def non_max_suppression(points, threshold=10):
    ''' Suppresses redundant keypoints based on threshold '''
    dm = distance_matrix(points, points, p=2)
    np.fill_diagonal(dm, 1e15)
    binarized = np.where(dm <= threshold, 1, 0)
    binarized = np.tril(binarized)
    unique_points = np.all(binarized != 1, axis=1)
    return points[unique_points]

def compute_sift_desc(keypoints1, keypoints2, image1, image2):
    # convert torch to numpy arrays
    keypoints1 = keypoints1.numpy()[0]
    keypoints2 = keypoints2.numpy()[0]
    image1 = np.uint8(image1.permute(0, 2, 3, 1).numpy() * 255)
    image2 = np.uint8(image2.permute(0, 2, 3, 1).numpy() * 255)

    # Create SIFT detector
    sift = cv2.SIFT_create()
    
    # Define your keypoints for both images (x, y coordinates)
    keypoints1 = [cv2.KeyPoint(x, y, 10) for x, y in keypoints1]
    keypoints2 = [cv2.KeyPoint(x, y, 10) for x, y in keypoints2]

    # Compute SIFT descriptors for the keypoints
    _, descriptors1 = sift.compute(image1[0], keypoints1)
    _, descriptors2 = sift.compute(image2[0], keypoints2)

    # convert to torch
    descriptors1 = torch.from_numpy(descriptors1)
    descriptors2 = torch.from_numpy(descriptors2)
    return torch.stack([descriptors1]), torch.stack([descriptors2])

def bf_matcher(descriptors1, descriptors2, thresh=0.9):
    # Create a BFMatcher (Brute-Force Matcher)
    bf = cv2.BFMatcher()
    # Match descriptors
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < thresh * n.distance:
            good_matches.append(m)
    return matches, good_matches

def flann_matcher(descriptors1, descriptors2, thresh=1):
    # Create a FlannBasedMatcher
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match descriptors using the FlannBasedMatcher
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < thresh * n.distance:
            good_matches.append(m)
    return matches, good_matches

# compute segmentation based keypoints
def compute_keypoints_seg(fixed_vessel, fixed_disk, moving_vessel, moving_disk, seg_thresh=128, kp_thresh=10):

    ''' Compute keypoints guided by vessel and disk segmentation '''
    
    # convert torch to numpy arrays
    fixed_vessel = fixed_vessel.permute(0, 2, 3, 1).numpy() * 255
    fixed_disk = fixed_disk.permute(0, 2, 3, 1).numpy() * 255
    moving_vessel = moving_vessel.permute(0, 2, 3, 1).numpy() * 255
    moving_disk = moving_disk.permute(0, 2, 3, 1).numpy() * 255

    b = fixed_vessel.shape[0]

    fixed_keypoints_all = []
    moving_keypoints_all = []

    for i in range(b):
        # ==================
        # Keypoint detection
        # ==================
            
        # get intersection keypoints
        fixed_keypoints, moving_keypoints = get_intersection(
            fixed_disk[i, :, :, 0], 
            fixed_vessel[i, :, :, 0], 
            moving_disk[i, :, :, 0], 
            moving_vessel[i, :, :, 0], 
            thresh=seg_thresh
            )

        # filter out redundant keypoints
        fixed_keypoints = non_max_suppression(fixed_keypoints, threshold=kp_thresh).astype(np.float32)
        moving_keypoints = non_max_suppression(moving_keypoints, threshold=kp_thresh).astype(np.float32)

        # switch x and y
        fixed_keypoints[:, [0, 1]] = fixed_keypoints[:, [1, 0]]
        moving_keypoints[:, [0, 1]] = moving_keypoints[:, [1, 0]]

        fixed_keypoints_all.append(torch.tensor(fixed_keypoints))
        moving_keypoints_all.append(torch.tensor(moving_keypoints))

    return torch.stack(fixed_keypoints_all), torch.stack(moving_keypoints_all)

def compute_keypoints_loftr(fixed_image, moving_image, device='cpu'):
    ''' Computes the corresponding landmarks using keypoint model '''

    # load keypoint detector model
    model = KF.LoFTR(pretrained="outdoor").to(device)

    # prepare model input
    input_dict = {
    "image0": K.color.rgb_to_grayscale(fixed_image.to(device)),  # LofTR works on grayscale images only
    "image1": K.color.rgb_to_grayscale(moving_image.to(device)),
    }

    # run inference with kp model
    with torch.inference_mode():
        correspondences = model(input_dict)

    return correspondences

detectors = {
    'loftr': compute_keypoints_loftr,
    'seg': compute_keypoints_seg,
    'superpoint': SuperPoint(max_num_keypoints=2048).eval()
}

descriptors = {
    'loftr': compute_keypoints_loftr,
    'sift_1': compute_sift_desc,
    'sift_2': SIFT(),
    'superpoint': SuperPoint(max_num_keypoints=2048).eval()
}

matchers = {
    'loftr': compute_keypoints_loftr,
    'lightglue_sift': LightGlue(features="sift").eval(),
    'lightglue_superpoint': LightGlue(features="superpoint").eval(),
    'bf': bf_matcher,
    'flann': flann_matcher
}

def get_keypoints(fixed_image, moving_image, fixed_vessel, fixed_disk, moving_vessel, moving_disk, kp_method='seg', desc_method='sift', match_method='lightglue', device='cpu', inp='img', mask=None, top=None):
    ''' Detects and matches keypoints in images '''
    
    # load keypoint detector
    try:
        detector = detectors[kp_method].to(device)
    except:
        detector = detectors[kp_method]
    
    # load descriptor
    try:
        descriptor = descriptors[desc_method].to(device)
    except:
        descriptor = descriptors[desc_method]   

    # load matcher
    try:
        matcher = matchers[match_method].to(device)
    except:
        matcher = matchers[match_method]  

    timings = []

    # compute superpoint keypoints+features
    if (kp_method == 'superpoint' or desc_method == 'superpoint'):
        # Start the timer for this iteration
        start_time = time.time()

        # contains both the keypoints and descriptors
        if inp == 'vmask':
            # preprocess inputs
            fixed_vessel = (fixed_vessel > 0.5).float()
            moving_vessel = (moving_vessel > 0.5).float()

            fixed_inputs = descriptor.extract(fixed_vessel.to(device))
            moving_inputs = descriptor.extract(moving_vessel.to(device))

        elif inp == 'dmask':
            # preprocess disk inputs
            fixed_disk_mask = 1 - (fixed_disk > 0.5).float()
            moving_disk_mask = 1 - (moving_disk > 0.5).float()

            fixed_inputs = descriptor.extract(fixed_disk.to(device))
            moving_inputs = descriptor.extract(moving_disk.to(device))

        elif inp == 'structural':
            # preprocess vessel inputs
            fixed_vessel = (fixed_vessel > 0.5).float()
            moving_vessel = (moving_vessel > 0.5).float()

            # preprocess disk inputs
            fixed_disk_mask = 1 - (fixed_disk > 0.5).float()
            moving_disk_mask = 1 - (moving_disk > 0.5).float()

            fixed_mask = fixed_vessel * fixed_disk_mask
            moving_mask = moving_vessel * moving_disk_mask

            fixed_inputs = descriptor.extract(fixed_mask.to(device))
            moving_inputs = descriptor.extract(moving_mask.to(device))
        
        else:
            fixed_inputs = descriptor.extract(fixed_image.to(device))
            moving_inputs = descriptor.extract(moving_image.to(device))

        # Start the timer for this iteration
        end_time = time.time()
        timings += [end_time - start_time]

    if (kp_method == 'loftr' or match_method == 'loftr' or match_method == 'loftr'):
        # Start the timer for this iteration
        start_time = time.time()

        # contains both the keypoints and descriptors
        if inp == 'vmask':
            # preprocess inputs
            fixed_vessel = (fixed_vessel > 0.5).float()
            moving_vessel = (moving_vessel > 0.5).float()
            correspondences = compute_keypoints_loftr(fixed_vessel.to(device), moving_vessel.to(device), device=device) # (N, 2)
        
        elif inp == 'dmask':
            # preprocess disk inputs
            fixed_disk_mask = 1 - (fixed_disk > 0.5).float()
            moving_disk_mask = 1 - (moving_disk > 0.5).float()
            correspondences = compute_keypoints_loftr(fixed_disk.to(device), moving_disk.to(device), device=device) # (N, 2)
        
        elif inp == 'structural':
            # preprocess vessel inputs
            fixed_vessel = (fixed_vessel > 0.5).float()
            moving_vessel = (moving_vessel > 0.5).float()

            # preprocess disk inputs
            fixed_disk_mask = 1 - (fixed_disk > 0.5).float()
            moving_disk_mask = 1 - (moving_disk > 0.5).float()

            fixed_mask = fixed_vessel * fixed_disk_mask
            moving_mask = moving_vessel * moving_disk_mask

            correspondences = compute_keypoints_loftr(fixed_mask.to(device), moving_mask.to(device), device=device) # (N, 2)

        else:
            correspondences = compute_keypoints_loftr(fixed_image.to(device), moving_image.to(device), device=device) # (N, 2)

        # Start the timer for this iteration
        end_time = time.time()
        timings += [end_time - start_time]

    # extract keypoints from image
    if kp_method == 'superpoint':
        fixed_keypoints, moving_keypoints = fixed_inputs['keypoints'], moving_inputs['keypoints']
    elif kp_method == 'loftr':
        # extract keypoints with confidences
        fixed_keypoints = correspondences["keypoints0"].cpu()
        moving_keypoints = correspondences["keypoints1"].cpu()
    else: 
        fixed_keypoints, moving_keypoints = detector(fixed_vessel, fixed_disk, moving_vessel, moving_disk)

    # compute descriptors for keypoints
    if desc_method == 'superpoint':
        fixed_features = fixed_inputs['descriptors']
        moving_features = moving_inputs['descriptors']
    elif desc_method == 'loftr':
        pass
    else:
        if 'lightglue' in match_method:
            fixed_features, moving_features = descriptor()
        else:
            fixed_features, moving_features = descriptor(fixed_keypoints, moving_keypoints, fixed_image, moving_image)

    # match keypoint descriptors
    if 'lightglue' in match_method:
        # Start the timer for this iteration
        start_time = time.time()

        fixed_data = {
            'keypoints': fixed_keypoints,
            'descriptors': fixed_features
        }
        moving_data = {
            'keypoints': moving_keypoints,
            'descriptors': moving_features
        }
        matches01 = matcher({"image0": fixed_data, "image1": moving_data}) 
        matches = matches01["matches"][0]
        scores = matches01["scores"][0]

        fixed_keypoints, moving_keypoints = fixed_keypoints[0][matches[..., 0]], moving_keypoints[0][matches[..., 1]]

        # Start the timer for this iteration
        end_time = time.time()
        timings += [end_time - start_time]

    elif 'loftr' in match_method:
        scores = correspondences['confidence'].cpu()
    else:
        matches, good_matches = matcher(fixed_features[0].numpy(), moving_features[0].numpy())
        if match_method == 'bf' or match_method == 'flann':
            fixed_keypoints = np.array([fixed_keypoints[0][match.queryIdx] for match in good_matches])
            moving_keypoints = np.array([moving_keypoints[0][match.trainIdx] for match in good_matches])
        else:
            fixed_keypoints, moving_keypoints = fixed_keypoints[good_matches], moving_keypoints[matches[good_matches]]
        fixed_keypoints, moving_keypoints = torch.from_numpy(fixed_keypoints), torch.from_numpy(moving_keypoints)

    if mask is not None:
        # Start the timer for this iteration
        start_time = time.time()

        if mask == 'vmask':
            # threshold masks
            fixed_mask = np.uint8((fixed_vessel.permute(0, 2, 3, 1).numpy()[0, :, :, 0] * 255) > 128)
            moving_mask = np.uint8((moving_vessel.permute(0, 2, 3, 1).numpy()[0, :, :, 0] * 255) > 128)

        elif mask == 'dmask':

            # threshold masks
            fixed_mask = np.uint8(np.array(Image.open('data/retina_datasets/FIRE/Masks/mask.png').resize((256, 256))) > 128)
            moving_mask = np.uint8(np.array(Image.open('data/retina_datasets/FIRE/Masks/mask.png').resize((256, 256))) > 128)

            fixed_disk_mask = np.uint8(((1 - fixed_disk).permute(0, 2, 3, 1).numpy()[0, :, :, 0] * 255) > 128)
            moving_disk_mask = np.uint8(((1 - moving_disk).permute(0, 2, 3, 1).numpy()[0, :, :, 0] * 255) > 128)

            fixed_mask = fixed_mask * fixed_disk_mask
            moving_mask = moving_mask * moving_disk_mask

        elif mask == 'structural':
            # threshold masks
            fixed_vessel_mask = torch.clip(fixed_vessel * (1 - fixed_disk), min=0, max=1)
            moving_vessel_mask = torch.clip(moving_vessel * (1 - moving_disk), min=0, max=1)
            fixed_vessel = np.uint8((fixed_vessel_mask.permute(0, 2, 3, 1).numpy()[0, :, :, 0] * 255) > 128)
            moving_vessel = np.uint8((moving_vessel_mask.permute(0, 2, 3, 1).numpy()[0, :, :, 0] * 255) > 128)

            fixed_disk_mask = cv2.distanceTransform(np.uint8(fixed_disk.permute(0, 2, 3, 1)[0, :, :, 0].numpy() * 255), cv2.DIST_C, 5)
            fixed_disk_mask = fixed_disk_mask == 1
            moving_disk_mask = cv2.distanceTransform(np.uint8(moving_disk.permute(0, 2, 3, 1)[0, :, :, 0].numpy() * 255), cv2.DIST_C, 5)
            moving_disk_mask = moving_disk_mask == 1

            fixed_mask = np.clip(fixed_vessel + fixed_disk_mask, 0, 1)
            moving_mask = np.clip(moving_vessel + moving_disk_mask, 0, 1)

        # fixed keypoints
        fixed_keypoints_x = fixed_keypoints[:, 0] # (N,2) -> (N,) 
        fixed_keypoints_y = fixed_keypoints[:, 1] # (N,)

        # moving keypoints
        moving_keypoints_x = moving_keypoints[:, 0] # (N,)
        moving_keypoints_y = moving_keypoints[:, 1] # (N,)

        # filter keypoints
        fmask = fixed_mask[fixed_keypoints_y.int(), fixed_keypoints_x.int()]
        mmask = moving_mask[moving_keypoints_y.int(), moving_keypoints_x.int()]
        mask = torch.tensor([a*b for a,b in zip(fmask, mmask)]).bool()

        scores_filtered = scores[mask]
        fixed_keypoints_filtered = fixed_keypoints[mask]
        moving_keypoints_filtered = moving_keypoints[mask]

        # Start the timer for this iteration
        end_time = time.time()
        timings += [end_time - start_time]

        # Start the timer for this iteration
        start_time = time.time()

        # select top100 most confident point matches
        if top is not None:
            confidences = scores_filtered
            top_N = torch.argsort(confidences, descending=True)[:min(top, len(confidences))]
            fixed_keypoints_filtered = fixed_keypoints_filtered[top_N]
            moving_keypoints_filtered = moving_keypoints_filtered[top_N]

        # Start the timer for this iteration
        end_time = time.time()
        timings += [end_time - start_time]

    else:
        fixed_keypoints_filtered = fixed_keypoints
        moving_keypoints_filtered = moving_keypoints

        # Start the timer for this iteration
        start_time = time.time()
        # select top100 most confident point matches
        if top is not None:        
            confidences = scores
            top_N = torch.argsort(confidences, descending=True)[:min(top, len(confidences))]
            fixed_keypoints_filtered = fixed_keypoints_filtered[top_N]
            moving_keypoints_filtered = moving_keypoints_filtered[top_N]

        # Start the timer for this iteration
        end_time = time.time()
        timings += [end_time - start_time]

    return torch.stack([fixed_keypoints_filtered]), torch.stack([moving_keypoints_filtered]), torch.stack([fixed_keypoints]), torch.stack([moving_keypoints])

def get_keypoints_splg(fixed_image, moving_image):
    ''' Detects and matches keypoints in images '''

    # load kp detector
    kp_detector = detectors['superpoint'].to(fixed_image.device)
    matcher = matchers['lightglue_superpoint'].to(fixed_image.device)

    # run inference
    fixed_inputs = kp_detector.extract(fixed_image)
    moving_inputs = kp_detector.extract(moving_image)

    # extract keypoints from image
    fixed_keypoints, moving_keypoints = fixed_inputs['keypoints'], moving_inputs['keypoints']

    # compute descriptors for keypoints
    fixed_features, moving_features = fixed_inputs['descriptors'], moving_inputs['descriptors']

    # match keypoint descriptors
    fixed_data = {
        'keypoints': fixed_keypoints,
        'descriptors': fixed_features
    }
    moving_data = {
        'keypoints': moving_keypoints,
        'descriptors': moving_features
    }
    matches01 = matcher({"image0": fixed_data, "image1": moving_data}) 
    matches = matches01["matches"][0]
    scores = matches01["scores"][0]
    fixed_keypoints, moving_keypoints = fixed_keypoints[0][matches[..., 0]], moving_keypoints[0][matches[..., 1]]
    return torch.stack([fixed_keypoints]), torch.stack([moving_keypoints])

def get_keypoints_loftr(fixed_image, moving_image):
    device = fixed_image.device
    correspondences = compute_keypoints_loftr(fixed_image.unsqueeze(0), moving_image.unsqueeze(0), device)
    fixed_keypoints = correspondences["keypoints0"]
    moving_keypoints = correspondences["keypoints1"]
    scores = correspondences['confidence'].cpu()
    return torch.stack([fixed_keypoints]), torch.stack([moving_keypoints])