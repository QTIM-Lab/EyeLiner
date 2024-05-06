''' Visualization functions '''

import sys
import numpy as np
import torch
from patchify import patchify, unpatchify
from PIL import Image, ImageDraw
import random

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

def visualize_deformation_grid(sampling_grid, image_shape, step_size=10, save_to=None):
    """
    Visualizes the deformation field represented by the sampling grid.

    Args:
    - sampling_grid (torch.Tensor): A PyTorch tensor representing the sampling grid with normalized coordinates.
    - image_shape (tuple): Shape of the input image.
    - step_size (int): Step size for displaying arrows.

    Returns:
    - deformation_field (np.ndarray): Array representing the deformation field.
    """
    # Convert normalized coordinates to image coordinates
    normalized_grid = sampling_grid.numpy()  # [height, width, 2]

    grid_x = np.linspace(0, 256, image_shape[1])
    grid_y = np.linspace(0, 256, image_shape[0])
    deformed_x = (normalized_grid[..., 0] + 1) * (image_shape[0] - 1) / 2
    deformed_y = (normalized_grid[..., 1] + 1) * (image_shape[1] - 1) / 2

    # Plot deformation field with specified step size
    plt.figure(figsize=(8, 8))
    # plt.imshow(np.ones(image_shape), cmap='gray')  # Show blank canvas
    plt.quiver(grid_x[::step_size], grid_y[::step_size], deformed_x[::step_size, ::step_size] - grid_x[::step_size, None], deformed_y[::step_size, ::step_size] - grid_y[::step_size, None], scale=1, scale_units='xy', angles='xy', color='r')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.title('Deformation Field')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.savefig(save_to)
    plt.close()