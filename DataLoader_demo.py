import torch
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Import the CustomCollate and DataLoader classes from the DataLoader module
from DataLoader import CustomCollate, DataLoader

###############################################################################
######### Check CUDA device availability and print device information #########
###############################################################################

# Check CUDA device availability and print device information
separator = '-' * 76
label = 'CUDA'
print(separator)
print(f'{label:^76}')
print(separator)
print()

if torch.cuda.is_available():
    # Get the current CUDA device and its name
    device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(device)
    print(f"Using CUDA device: {device_name}")
    print(f"Memory Usage: {torch.cuda.memory_allocated(device) / 1024 ** 3} GB")

###############################################################################
########################## Calling the DataLoader #############################
###############################################################################

# Print a label to indicate the Data Loader section
label = 'Data Loader'
print()
print(separator)
print(f'{label:^76}')
print(separator)
print()

# Get the root directory for the dataset
root_dir = os.getcwd() + '/train_dir/'

# Create a DataLoader instance using the root directory
dataset = DataLoader(root_dir)

# Create a dataloader using the DataLoader instance and CustomCollate function
dataloader = torch.utils.data.DataLoader(dataset, collate_fn=CustomCollate())

# Initialize lists to store potential energies and crack tips for all iterations
all_potential_energies = []
all_crack_tips = []

# Initialize a counter to number the iterations
iteration_counter = 1

for x, y in dataloader:
    # Display iteration information for y
    print(f"Iteration {iteration_counter}:")
    
    # Get the list of iteration folders in the train_dir
    train_dir = dataloader.dataset.root_dir
    iteration_folders = [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]
    current_iteration = iteration_folders[iteration_counter - 1]
    print(f"Iteration Name: {current_iteration}")
    
    # Get the number of crack tips in the current iteration
    num_crack_tips = y.shape[0]
    print(f"Number of Crack Tips in Initial Configuration (y): {num_crack_tips}")
    
    # Calculate the number of cracks (assuming each tip represents one crack)
    num_cracks = num_crack_tips // 4  # Integer division
    print(f"Number of Cracks in Initial Configuration (y): {num_cracks:.0f}")

    # Append the potential energy and crack tips to the corresponding lists
    all_potential_energies.append(x)
    all_crack_tips.append(y)
    
    iteration_counter += 1
    print("\n")

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot all the potential energies of all iterations
axs[0].set_title("Potential Energies of All Iterations")
for i, energy in enumerate(all_potential_energies):
    axs[0].plot(energy, label=f"Iteration {i + 1}")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Potential Energy")
axs[0].legend()

# Plot all the crack tips of all iterations
axs[1].set_title("Crack Tips of All Iterations")
for i, tips in enumerate(all_crack_tips):
    # Extract the coordinates for the beginning and end of each crack
    x_begin = tips[::4]
    y_begin = tips[1::4]
    x_end = tips[2::4]
    y_end = tips[3::4]
    
    # Connect the beginning and end points to form crack lines
    for x_start, y_start, x_stop, y_stop in zip(x_begin, y_begin, x_end, y_end):
        axs[1].plot([x_start, x_stop], [y_start, y_stop], marker="o", label=f"Iteration {i + 1}")
axs[1].set_xlabel("X Coordinate")
axs[1].set_ylabel("Y Coordinate")
axs[1].legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()