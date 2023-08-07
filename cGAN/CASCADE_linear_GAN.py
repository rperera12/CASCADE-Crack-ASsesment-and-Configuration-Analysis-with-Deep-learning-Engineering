import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from torchvision.utils import save_image

from DataLoader.DataLoader_3 import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(201, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256*256),
            nn.Tanh()
        )
        
    def forward(self, z, x):
        x = x.view(x.size(0), -1).float()
        input = torch.cat([z, x], dim=1)
        output = self.main(input)
        output = output.view(output.size(0), 1, 256, 256)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(256*256 + 101, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, y, x):
        x = x.view(x.size(0), -1).float()
        y = y.view(y.size(0), -1).float()
        input = torch.cat([y, x], dim=1)
        output = self.main(input)
        return output


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)

def train_discriminator(optimizer, real_data, fake_data, x):
    optimizer.zero_grad()

    # Trains on Real Data
    prediction_real = discriminator(real_data, x)
    error_real = criterion(prediction_real, torch.ones(real_data.size(0), 1).to(device))
    error_real.backward()

    # Trains on Fake Data
    prediction_fake = discriminator(fake_data, x)
    error_fake = criterion(prediction_fake, torch.zeros(fake_data.size(0), 1).to(device))
    error_fake.backward()

    optimizer.step()

    return error_real + error_fake

def train_generator(optimizer, fake_data, x):
    optimizer.zero_grad()
    
    prediction = discriminator(fake_data, x)
    error = criterion(prediction, torch.ones(fake_data.size(0), 1).to(device))
    error.backward()

    optimizer.step()

    return error

batch_size = 32  # You can change this to your desired batch size

train_dataset = CustomImageDataset(y_img_relative_path='../y_img', train_dir_relative_path='../train_dir', batch_size=batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for i, data in enumerate(train_loader):
    x, y = data
    print(f"Batch {i}:")
    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    break  # Breaks after printing the first batch

num_epochs = 100 # number of epochs to train for

# creates a folder to save generated images
if not os.path.exists('output_images'):
    os.makedirs('output_images')

# Initial setup for the live plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.set_title('Generated Images')
ax2.set_title('Losses')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
g_losses = []
d_losses = []

# path to the 'y_img' directory
y_img_directory = 'y_img/'

# Gets a list of files in the 'y_img' directory
file_list = os.listdir(y_img_directory)


image_extension = '.png'

# Checks if the 'y_img' directory contains any files
if len(file_list) > 0:
    # Sorts the file list to ensure the first image is selected
    file_list.sort()
    
    # Gets the path to the first image
    first_image_path = os.path.join(y_img_directory, file_list[0])

    # Reads the first image using OpenCV
    first_image = cv2.imread(first_image_path)
else:
    print("No images found in the 'y_img' directory.")

for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        
        batch_size = x.size(0)
        
        # creates the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        #============= TRAIN THE DISCRIMINATOR =============#
        z = torch.randn(batch_size, 100).to(device) # randomly generated noise
        generated_images = generator(z, x) # creates fake images
        
        d_loss_real = train_discriminator(d_optimizer, y, y, x) # real images
        d_loss_fake = train_discriminator(d_optimizer, y, generated_images.detach(), x) # fake images
        
        # sums up both losses
        d_loss = d_loss_real + d_loss_fake
        
        #============= TRAIN THE GENERATOR =============#
        # generates fake images
        z = torch.randn(batch_size, 100).to(device)
        generated_images = generator(z, x)
        
        # computes the loss for the generator
        g_loss = train_generator(g_optimizer, generated_images, x)

        d_error = train_discriminator(d_optimizer, y, generated_images.detach(), x)
        d_losses.append(d_error.item())

        # Trains Generator
        generated_images = generator(z, x)
        g_error = train_generator(g_optimizer, generated_images, x)
        g_losses.append(g_error.item())

        # Plotting logic
        if batch_idx % 100 == 0:  # Update plot every 100 batches
            ax1.clear()
            ax2.clear()

            # Shows first image (targeted image) on top
            ax1.imshow(cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB))

            # Shows generated image (live generating image) at the bottom
            generated_img = generated_images[0].squeeze().cpu().detach()  # Selects the first image from the batch
            ax1.imshow(generated_img, cmap='gray', alpha=0.5)

            # Shows losses
            ax2.plot(range(len(d_losses)), d_losses, label='Discriminator')
            ax2.plot(range(len(g_losses)), g_losses, label='Generator')
            ax2.legend()

            plt.pause(0.001)  # Small pause to allow the plots to update

    print(f"Epoch {epoch}: g_error={g_error}, d_error={d_error}")
    
    # prints some loss stats
    if epoch % 10 == 0:
        print(f'Epoch [{epoch}/{num_epochs}] | d_loss: {d_loss:.4f} | g_loss: {g_loss:.4f}')
        
        save_image(generated_images.view(generated_images.size(0), 1, 256, 256), f'output_images/sample_{epoch}.png')