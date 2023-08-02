import os 
import torch
import numpy as np
from torch import nn, optim
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, r2_score

from torch.utils.data import DataLoader
from DataLoader.DataLoader_3 import CustomImageDataset
from DataLoader.DataLoader_1 import CustomCollate, DataLoader_X

class Net(nn.Module):
    def __init__(self, neurons=512):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(101, neurons)  
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, neurons)
        self.fc4 = nn.Linear(neurons, neurons)
        self.dropout = nn.Dropout(0.5)  
        self.fc5 = nn.Linear(neurons, 256*256*1)  

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x.view(-1, 256, 256, 1)

# Defines learning rates and neurons for grid search
learning_rate = [0.1]
neurons = [128]

# Defines hyperparameters
batch_size = 64  # Batch size
epoch = 100

# Gets the root directory for the generated images
script_dir = os.path.dirname(os.path.realpath(__file__))
dataset_directory = os.path.abspath(os.path.join(script_dir, '..', 'y_img'))
root_dir = os.path.join(script_dir, '..', 'train_dir')

x_values = DataLoader_X(root_dir)

dataset = CustomImageDataset(root_dir=dataset_directory, x_dataset=x_values)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = Net().to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss

results = {}  # Stores final losses
losses = {}  # Stores losses at each epoch

# Grid search
# Initialize dictionaries to store metrics for each epoch
losses = {}
rmse_scores = {}
mae_scores = {}
r2_scores = {}

# Grid search
for lr in learning_rate:
    for neuron in neurons:
        model = Net(neuron).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_function = nn.MSELoss()

        # Initialize lists to store metrics for each epoch
        losses[(lr, neuron)] = []
        rmse_scores[(lr, neuron)] = []
        mae_scores[(lr, neuron)] = []
        r2_scores[(lr, neuron)] = []

        for epoch in range(epoch):
            epoch_predictions = []
            epoch_targets = []

            for inputs, targets in dataloader: 
                inputs, targets = inputs.float().to(device), targets.float().to(device)
                model.zero_grad() 
                outputs = model(inputs) 
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

                # Store predictions and targets for metrics calculation
                epoch_predictions.extend(outputs.detach().cpu().numpy().ravel())
                epoch_targets.extend(targets.detach().cpu().numpy().ravel())

            # Calculate metrics for this epoch
            epoch_rmse = np.sqrt(loss.item())  # Assuming that loss is MSE
            epoch_r2_score = r2_score(epoch_targets, epoch_predictions)

            # Store metrics for this epoch
            losses[(lr, neuron)].append(loss.item())
            rmse_scores[(lr, neuron)].append(epoch_rmse)
            r2_scores[(lr, neuron)].append(epoch_r2_score)

            # Print the metrics for this epoch, along with current learning rate and neurons
            print(f"Epoch: {epoch}, LR: {lr}, Neurons: {neuron}, Loss (MSE): {loss.item()}, RMSE: {epoch_rmse}, R2 Score: {epoch_r2_score}")

        results[(lr, neuron)] = loss.item()

# Print final losses for each configuration
for config, loss in results.items():
    print(f"Configuration: {config}, Final Loss: {loss}")

# Plotting the metrics
fig, axs = plt.subplots(2, 2, figsize=(15, 15))

# Plotting Loss (MSE)
for (lr, neuron), loss in losses.items():
    axs[0, 0].plot(loss, label=f"lr={lr}, neurons={neuron}")
axs[0, 0].set_title("Loss Mean Squared Error(MSE) over Epochs")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Loss (MSE)")
axs[0, 0].legend()

# Plotting RMSE
for (lr, neuron), rmse in rmse_scores.items():
    axs[0, 1].plot(rmse, label=f"lr={lr}, neurons={neuron}")
axs[0, 1].set_title("Root Mean Squared Error(RMSE) over Epochs")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("RMSE")
axs[0, 1].legend()

# Plotting MAE
for (lr, neuron), mae in mae_scores.items():
    axs[1, 0].plot(mae, label=f"lr={lr}, neurons={neuron}")
axs[1, 0].set_title("Mean Squared Error(MAE) over Epochs")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("MAE")
axs[1, 0].legend()

# Plotting R2 Scores
for (lr, neuron), r2_score in r2_scores.items():
    axs[1, 1].plot(r2_score, label=f"lr={lr}, neurons={neuron}")
axs[1, 1].set_title("R2 Scores over Epochs")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("R2 Score")
axs[1, 1].legend()

plt.tight_layout()
plt.show()

"""
Why are we not calculating or plotting the accuracy?
For a regression task, accuracy is not the appropriate metric, because accuracy measures the number 
of correct categorical predictions and does not take into account how close the predictions are to 
the actual values. In regression problems, we are more concerned with how close our predictions are 
to the actual values, rather than whether they are exactly correct.

The most common metrics for regression tasks include Mean Squared Error (MSE), Root Mean Squared 
Error (RMSE), Mean Absolute Error (MAE), and R2 score.

https://medium.com/analytics-vidhya/mae-mse-rmse-coefficient-of-determination-adjusted-r-squared-which-metric-is-better-cd0326a5697e

"""

# Select a random sample from the training dataset
random_index = np.random.randint(len(dataset))
random_input, random_target = dataset[random_index]

# Convert the random input to a tensor and move it to the device
random_input = torch.tensor(random_input).float().unsqueeze(0).to(device)

# Pass the input through the trained model to obtain the predicted output
with torch.no_grad():
    model.eval()  # Set the model to evaluation mode (important for dropout layers, if any)
    predicted_output = model(random_input)
    model.train()  # Set the model back to training mode

# Convert output tensor to numpy array and reshape
predicted_output_np = predicted_output.cpu().numpy().reshape(256, 256)

# Display the predicted output
plt.imshow(predicted_output_np, cmap='gray')
plt.show()

all_predicted_outputs = []
all_ground_truth = []

# Evaluate the model on the training dataset
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for x, y in dataloader:
        # Move the input (x) to the device
        x = x.float().to(device)
        # Pass the input through the model to obtain the predicted output
        predicted_output = model(x)
        # Move the predicted output back to the CPU and append to the list
        all_predicted_outputs.append(predicted_output.cpu().numpy())
        # Append the ground truth (y) to the list
        all_ground_truth.append(y.numpy())

# Create a figure for the plot
plt.figure(figsize=(8, 6))
plt.title("Predicted Output vs Ground Truth")
plt.xlabel("x coordinate")
plt.ylabel("y coordinate")

# Define the index of the sample you want to plot
index_to_plot = 0 

# Extract the predicted output and ground truth for the desired sample
predicted_output = all_predicted_outputs[0][index_to_plot]
ground_truth = all_ground_truth[0][index_to_plot]

# Reshape outputs to (5, 4) - each row will contain [x_beginning, y_beginning, x_end, y_end] for a crack
predicted_output = predicted_output.reshape(5, 4)
ground_truth = ground_truth.reshape(5, 4)

# Plot predicted and ground truth cracks
for i in range(5): # For five cracks
    plt.plot([predicted_output[i, 0], predicted_output[i, 2]], [predicted_output[i, 1], predicted_output[i, 3]], label=f"Predicted Crack {i+1}")
    plt.plot([ground_truth[i, 0], ground_truth[i, 2]], [ground_truth[i, 1], ground_truth[i, 3]], label=f"Ground Truth Crack {i+1}", linestyle="--")

# Adjust layout and display the plot
plt.legend()
plt.tight_layout()
plt.show()
