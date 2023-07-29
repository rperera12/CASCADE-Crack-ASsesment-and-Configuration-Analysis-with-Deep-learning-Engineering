import torch
import os
import scipy.io
import numpy as np

class CustomCollate:
    def __call__(self, batch):
        x = torch.stack([item[0].flatten() for item in batch])
        y = torch.stack([item[1].flatten() for item in batch])
        return x, y

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, root_dir, pytorch=True, normalize=True):
        super().__init__()
        self.root_dir = root_dir
        self.pytorch = pytorch
        self.normalize = normalize
        self.n_iter = self.get_available_iterations()
        if self.normalize:
            self.global_max = self.get_global_max()

    def get_global_max(self):
        global_max = None
        for i in range(len(self.n_iter)):
            current_iter = self.n_iter[i]
            state_file = os.path.join(self.root_dir, f'var_iter_{current_iter}', 'var_State.mat')
            es_data = scipy.io.loadmat(state_file)
            sState = es_data['sState']
            es = sState['Es']
            es_nested_array = es[0, 0]
            es_values = es_nested_array[:, 0]
            if global_max is None or np.max(es_values) > global_max:
                global_max = np.max(es_values)
        return global_max

    def get_available_iterations(self):
        files = os.listdir(self.root_dir)
        n_iter = [int(file.split('_')[2]) for file in files if file.startswith('var_iter_')]
        return sorted(n_iter)

    def __len__(self):
        if len(self.n_iter) == 0:
            print("Warning! The length is 0")
        return len(self.n_iter)
    
    def get_crack_tip_positions(self, crack_tip_positions):
        positions = []
        for item in crack_tip_positions:
            tip_positions = item[0]
            positions.extend(tip_positions.flatten())

        positions = np.array(positions, dtype=np.float32)

        crack_positions = torch.tensor(positions, dtype=torch.float32)

        return crack_positions
    
    def normalize_data(self, data, norm_value=None):
        data = data.clone().detach()
        if norm_value is None:  # If no normalization value is provided, uses global max
            norm_value = self.global_max
        data_normalized = data / norm_value
        return data_normalized
    
    def pad_tensor(self, data, target_length=101):
        # Converts data to PyTorch tensor if it isn't already
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        # If the data is already the target length, return it as is
        if data.numel() == target_length:
            return data

        # If the data is longer than the target length, truncate it
        if data.numel() > target_length:
            return data[:target_length]

        # If the data is shorter than the target length, pad it with zeros
        if data.numel() < target_length:
            return torch.cat([data, torch.zeros(target_length - data.numel())])

    def __getitem__(self, index):
        current_iter = self.n_iter[index]
        initial_conf_file = os.path.join(self.root_dir, f'var_iter_{current_iter}', 'var_Crack_initial.mat')
        state_file = os.path.join(self.root_dir, f'var_iter_{current_iter}', 'var_State.mat')
        initial_cracks = scipy.io.loadmat(initial_conf_file)
        es_data = scipy.io.loadmat(state_file)

        try:
            sState = es_data['sState']
            es = sState['Es']
            es_nested_array = es[0, 0]
            es_values = es_nested_array[:, 0]
        except (KeyError, IndexError) as e:
            raise ValueError("Error accessing potential energy data. Check the structure of the loaded MATLAB file.") from e
        try:
            crack_id = initial_cracks['cCkCrd']
            num_cracks = len(crack_id)
            crack_tip_positions = []

            for i in range(num_cracks):
                tip_positions = crack_id[i]
                tip_positions = np.squeeze(tip_positions)
                crack_tip_positions.append(tip_positions)

            crack_tip_positions = np.vstack(crack_tip_positions)
            crack_tip_positions = self.get_crack_tip_positions(crack_tip_positions)

        except (KeyError, IndexError) as e:
            raise ValueError("Error accessing crack ID. Check the structure of the loaded MATLAB file.") from e

         # Convert numpy arrays to tensors
        x = torch.from_numpy(es_values)
        y = crack_tip_positions

        # print(f'Max before normalization - x: {torch.max(x)}, y: {torch.max(y)}')
        if self.normalize:
            x = self.normalize_data(x)
            y = self.normalize_data(y, 2500)
        # print(f'Max after normalization - x: {torch.max(x)}, y: {torch.max(y)}')
        
        # Return the padded tensors
        return self.pad_tensor(x.flatten()), y.flatten()
    
###############################################################################
######### Check CUDA device availability and print device information #########
###############################################################################

# Ensures that all the test information will only be printed if DataLoader.py 
# is runned, not if its imported as a module.

if __name__ == "__main__":

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

    # Get the list of iteration folders in the train_dir
    train_dir = dataloader.dataset.root_dir
    iteration_folders = [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]

    print()
    print(separator)
    print(f"{'x = Potential Energy over Time':^75}")
    print(separator)
    print()

    # Initialize a counter to number the iterations
    iteration_counter = 1

    ###############################################################################
    ###################### Information [Potential Energy] #########################
    ###############################################################################

    for x, _ in dataloader:
        # Display iteration information
        print(f"Iteration {iteration_counter}:")
        
        # Get the list of iteration folders in the train_dir
        train_dir = dataloader.dataset.root_dir
        iteration_folders = [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]
        current_iteration = iteration_folders[iteration_counter - 1]
        print(f"Iteration Name: {current_iteration}")
        
        # Get the length of the 1D array for the current potential energy history (x)
        x_length = x.shape[-1]
        print(f"Length of Potential Energy History (x): {x_length}")
        
        iteration_counter += 1
        print("\n")

    # Plotting Initial Crack Configurations
    print()
    print(separator)
    print(f"{'y = Initial Crack Configurations':^75}")
    print(separator)
    print()

    # Reinitialize the iteration counter for y
    iteration_counter = 1

    ###############################################################################
    ###################### Information [Crack Positions] #########################
    ###############################################################################

    for _, y in dataloader:
        # Display iteration information for y
        print(f"Iteration {iteration_counter}:")
        
        # Get the list of iteration folders in the train_dir
        train_dir = dataloader.dataset.root_dir
        iteration_folders = [folder for folder in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, folder))]
        current_iteration = iteration_folders[iteration_counter - 1]
        print(f"Iteration Name: {current_iteration}")
        
        # Get the number of crack tips in the current iteration
        num_crack_tips = y.shape[-1]
        print(f"Number of Crack Tips in Initial Configuration (y): {num_crack_tips}")
        
        # Calculate the number of cracks (assuming each tip represents one crack)
        num_cracks = num_crack_tips // 4  # Integer division
        
        # Display the number of cracks without the decimal point when it's an integer
        print(f"Number of Cracks in Initial Configuration (y): {num_cracks:.0f}")
        
        iteration_counter += 1
        print("\n")

    for i, (x, y) in enumerate(dataloader):
        print(f"Iteration {i+1}:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print("\n")
