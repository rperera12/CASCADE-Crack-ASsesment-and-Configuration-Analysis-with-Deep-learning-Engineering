import torch
import os
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from DataLoader_1 import DataLoader_X

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, x_dataset, transform=None):
        self.root_dir = root_dir
        self.x_dataset = x_dataset
        self.transform = transform
        self.list_of_files = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]

    def __len__(self):
        return len(self.list_of_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.list_of_files[idx])
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        image = image[:, :, np.newaxis]  # Adds channel dimension

        if self.transform:
            image = self.transform(image)

        x = self.x_dataset[idx][0]  # Selects the first element of the tuple

        return x, image

transform = transforms.Compose([
    transforms.ToTensor(),  # Converts the image to PyTorch Tensor data type
])

# Get the root directory for the generated images
script_dir = os.path.dirname(os.path.realpath(__file__))
dataset_directory = os.path.abspath(os.path.join(script_dir, '..', 'y_img'))

root_dir = os.path.join(script_dir, '..', 'train_dir')

x_values = DataLoader_X(root_dir) 

dataset = CustomImageDataset(root_dir=dataset_directory, x_dataset=x_values, transform=transform)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)


# In case testing is needed, this will output some basic information about our x's and generated y's.

# Assuming your dataloader is named dataloader
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
