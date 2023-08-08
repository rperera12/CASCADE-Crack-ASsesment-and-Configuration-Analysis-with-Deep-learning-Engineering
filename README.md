# CASCADE: Crack ASsesment and Configuration Analysis with Deep-learning Engineering

## What is Cascade?

CASCADE is a project that leverages the power of Deep Learning, particularly Conditional Generative Adversarial Networks (cGANs) and Feed Forward Neural Networks (FFNNs), to assess and analyze crack configurations. It takes raw data from MATLAB files and transforms them into useful inputs for these networks (Potential Energy Distribution throghout time), which will produce images representing the corresponding initial crack configurations. 

## Last Modification

This script was last modified on: August 8, 2023

## Installation

Before installing CASCADE, make sure to have Python installed on your system. If not, you can download it from the [official Python website](https://www.python.org/downloads/). Additionally, CASCADE requires specific Python packages to function correctly, including tensorflow, matplotlib, numpy, and others.

To install CASCADE:

1. Clone the repository to your local machine using git:

\```bash
git clone https://github.com/rperera12/CASCADE-Crack-ASsesment-and-Configuration-Analysis-with-Deep-learning-Engineering.git
\```

2. Navigate to the downloaded directory:

\```bash
cd CASCADE
\```

3. Install the required packages using pip:

\```bash
pip install -r requirements.txt
\```

## Usage

To use CASCADE, make sure you are in the root directory of the project and then you can run any script by using the Python `-m` option followed by the folder and script name. For example:

\```bash
python -m cGAN.CASCADE_linear_GAN
\```

## Files Layout

The project follows the following directory structure:

- **`cGAN/`**: Contains the files that will produce the conditional GAN.
  - **`CASCADE_linear_GAN.py`**: Conditional GAN that produces images for the given Potential Energies.
- **`DataLoader/`**: Contains the DataLoaders that will produce the x and y tensors used to train the model.
  - **`DataLoader_1.py`**: Produces tensored x and y from the given MATLAB files.
  - **`DataLoader_2_PNG.py`**: Generates images from the Initial Crack Configurations.
  - **`DataLoader_3.py`**: Tensors the generated images and includes them in x.
- **`FFNN/`**: Contains the files that will produce the FFNN.
  - **`CASCADE_FFNN.py`**: FFNN that produces images for the given Potential Energies.
- **`train_dir/`**: Contains the raw training MATLAB files.
- **`y_img/`**: Contains the generated images for the Initial Crack Configurations.

## Note

To run the project, it must be run on the terminal:

\```bash
cd C:\Users\luism\OneDrive\Documents\Files\Research\CASCADE-Crack-ASsesment-and-Configuration-Analysis-with-Deep-learning-Engineering-Luis\
python -m folder.script
\```
Make sure to replace `folder.script` with the actual folder and script you intend to run. For instance, `python -m cGAN.CASCADE_linear_GAN`.
