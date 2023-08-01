# CASCADE: Crack ASsesment and Configuration Analysis with Deep-learning Engineering

## What is Cascade?

TBD

## Last Modification

This script was last modified on: <!-- Insert last modification date here -->

## Installation

TBD

## Usage

TBD

## Files Layout

The project follows the following directory structure:

- **`DataLoader/`**: Contains the DataLoaders that will produce the x and y tensors used to train the model.
  - **`DataLoader_1.py`**: Produces tensored x and y from the given MATLAB files.
  - **`DataLoader_2_PNG.py`**: Generates images from the Initial Crack Configurations.
  - **`DataLoader_3.py`**: Tensors the generated images and includes them in x.
- **`train_dir/`**: Contains the raw training MATLAB files.
- **`y_img/`**: Contains the generated images for the Initial Crack Configurations.
- **`CASCADE_FFNN.py`**: FFNN that recognizes the Crack Configurations for the given Potential Energies.
- **`CASCADE_cGAN.py`**: Conditional GAN that produces images for the given Potential Energies.
