import os
import torch.nn
from torch.utils.data import DataLoader

from DataLoader.DataLoader_3 import CustomImageDataset

main = os.getcwd()
print(main)

batch_size = 32  # You can change this to your desired batch size

y_img_path = os.path.abspath(main + '/y_img')  # get absolute path, this is useful for debugging

train_dataset = CustomImageDataset(y_img_relative_path=y_img_path, train_dir_relative_path='../train_dir', batch_size=batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for i, data in enumerate(train_loader):
    x, y = data
    print(f"Batch {i}:")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
