import numpy as np
import torch
from patchify import patchify, unpatchify

def extract_patches(image, patch_size):
    width, height = image.size
    patches = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(np.array(patch))
    return patches

def create_checkerboard(image1, image2, patch_size=8):

    C, H, W = image1.shape

    image1 = torch.permute(image1, (1, 2, 0)).detach().cpu().numpy()
    image2 = torch.permute(image2, (1, 2, 0)).detach().cpu().numpy()
    
    image1 = patchify(image1, (patch_size, patch_size, C), step=patch_size) # (n, n, 1, p, p, c)
    image2 = patchify(image2, (patch_size, patch_size, C), step=patch_size) # (n, n, 1, p, p, c)
    h, w, _, _, _, _ = image1.shape
    
    cb_image = image1.copy()

    for i in range(h):
        for j in range(w):   
            
            if (i % 2  ==  j % 2):
                # print(f'Black: {i, j}')
                cb_image[i, j] = image1[i, j]
            else:
                # print(f'White: {i, j}')
                cb_image[i, j] = image2[i, j]
                
    cb_image = unpatchify(cb_image, (H, W, C))
    # cb_image = Image.fromarray(np.uint8(cb_image))
    cb_image = torch.permute(torch.tensor(cb_image), (2, 0, 1))
    return cb_image
