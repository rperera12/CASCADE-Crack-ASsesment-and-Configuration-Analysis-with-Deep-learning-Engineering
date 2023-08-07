import torch
import os
import numpy as np
import matplotlib.pyplot as plt

"""
x = potential energy throught timestamps (101)
y = Initial crack configuration (5 cracks, 4 points for each (beginning of crack x and y, and end of crack x and y))
"""

# Imports the CustomCollate and DataLoader classes from the DataLoader module
from DataLoader_1 import CustomCollate, DataLoader_X


if torch.cuda.is_available():
    # Gets the current CUDA device and its name
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)

# Gets the root directory for the dataset
script_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, '..', 'train_dir'))

# Creates a DataLoader instance using the root directory
dataset = DataLoader_X(root_dir)

# Creates a dataloader using the DataLoader instance and CustomCollate function
dataloader = torch.utils.data.DataLoader(dataset, collate_fn=CustomCollate())

# Defines grid size
grid_size = 256

def bresenham_line(x1, y1, x2, y2):
    points = []
    dx = x2 - x1
    dy = y2 - y1

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        points.append((x1 + x*xx + y*yx, y1 + x*xy + y*yy))
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

    return points

def create_image_from_cracks(crack_config, grid_size, max_value=2500):
    # Initializes a grid of zeros
    image = np.zeros((grid_size, grid_size))

    # Scaling factor
    scale = grid_size / max_value

    # Reshapes crack_config into a 2D tensor
    crack_config = crack_config.view(-1, 4)

    for crack in crack_config:

        # Now, we assume that each 'crack' could represent multiple cracks, each represented by four coordinates
        for i in range(0, len(crack), 4):
            # Unnormalizes the data
            unnormalized_crack = np.array(crack[i:i+4]) * max_value
            
            scaled_crack = unnormalized_crack * scale
            x1, y1, x2, y2 = map(int, scaled_crack)

            # Uses Bresenham's line algorithm to generate a line between the start and end points
            points = bresenham_line(x1, y1, x2, y2)
            for point in points:
                image[point[1], point[0]] = 1

    # Transposes the image to fit with the standard image coordinate system
    image = np.transpose(image)

    return image

os.makedirs('y_img', exist_ok=True)  # Makes sure the directory exists

for i, (_, y_batch, iteration_names) in enumerate(dataloader):
    for j, y in enumerate(y_batch):

        # Gets the filename for this iteration
        iteration_name = iteration_names[j]
        img_path = os.path.join('y_img', iteration_name + '.png')

        # If the image already exists, skip this iteration
        if os.path.exists(img_path):
            continue

        image = create_image_from_cracks(y, grid_size)
    
        fig, ax = plt.subplots(figsize=(1,1), dpi=grid_size)
        ax.imshow(image, cmap='gray', aspect='auto')

        ax.axis('off')
        plt.grid(False)
        plt.tight_layout(pad=0)

        # Saves the image
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.close("all")

        if i == 0 and j == 0:  # Only showing the first one for testing purposes
            break

print("Done! All images available from the dataset created.")