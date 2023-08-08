import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from .DataLoader_1 import DataLoader_X
from .DataLoader_2_PNG import process_and_save_images  # Importing the new function

class CustomImageDataset(Dataset):
    def __init__(self, y_img_relative_path, train_dir_relative_path, batch_size, transform=None):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self.y_img_dir = os.path.abspath(os.path.join(script_dir, y_img_relative_path))
        train_dir = os.path.abspath(os.path.join(script_dir, train_dir_relative_path))
        self.x_dataset = DataLoader_X(train_dir)
        self.transform = transform
        
        # Check if y_img directory exists and contains files
        if not os.path.exists(self.y_img_dir) or not os.listdir(self.y_img_dir):
            print("y_img directory is empty or non-existent. Creating images using DataLoader_2_PNG.py...")
            process_and_save_images()  # Calling the function to generate and save images
            
        self.list_of_files = [f for f in os.listdir(self.y_img_dir) if os.path.isfile(os.path.join(self.y_img_dir, f))]
        self.batch_size = batch_size

    def __len__(self):
        return len(self.list_of_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.y_img_dir, self.list_of_files[idx])
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        y = image[:, :, np.newaxis]
        y = torch.from_numpy(y).float()

        if self.transform:
            y = self.transform(y)

        x = self.x_dataset[idx][0]
        x = x.unsqueeze(0)

        return x, y


"""

# Get the root directory for the generated images
script_dir = os.path.dirname(os.path.realpath(__file__))
dataset_directory = os.path.abspath(os.path.join(script_dir, '..', 'y_img'))

root_dir = os.path.join(script_dir, '..', 'train_dir')

x_values = DataLoader_X(root_dir) 

dataset = CustomImageDataset(root_dir=dataset_directory, x_dataset=x_values)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# In case testing is needed, this will output some basic information about our x's and generated y's.

for x, y in dataloader:
    print("x shape:", x.shape)  # This should print torch.Size([101])
    print("y shape:", y.shape)  # This should print something like torch.Size([1, 256, 256])

for i, (x, y) in enumerate(dataloader):
    print(f"Batch: {i+1}")
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    
    if i == 0:  # for the first batch only
        print("First x in batch:", x[0])
        print("Number of crack pixels in first y in batch:", y[0].nonzero().size(0))
    print()

    if i == 4:  # stop after 5 batches
        break
"""