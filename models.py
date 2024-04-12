import numpy as np
import torch
import torch.nn as nn

class IntermediateFusionFeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        super(IntermediateFusionFeedForwardNN, self).__init__()
        hidden_size1 = [dim // 2 for dim in input_size]
        hidden_size2 = np.sum(hidden_size1) // 2
        self.fc_modality_layers = [nn.Sequential(
            nn.Linear(input_dim, hid_dim),
        ).to(device) for input_dim, hid_dim in zip(input_size, hidden_size1)]
        self.fc2 = nn.Linear(np.sum(hidden_size1), hidden_size2)
        self.fc_final = nn.Linear(hidden_size2, output_size)
        self.float()

    def forward(self, features):
        # Pass each modality through its respective hidden layers
        x_hidden = [layer(x_modality) for layer, x_modality in zip(self.fc_modality_layers, features)]
        # Concatenate hidden representations
        x_concatenated = torch.cat(x_hidden, dim=-1)
        x = torch.relu(self.fc2(x_concatenated))  # Apply ReLU activation to output of second layer
        x = self.fc_final(x)  # Output of third layer
        return x

class LateFusionFeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        super(LateFusionFeedForwardNN, self).__init__()
        hidden_size = [dim // 2 for dim in input_size]
        self.fc_modality_layers = [nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
        ).to(device) for input_dim, hid_dim in zip(input_size, hidden_size)]
        self.fc_final = nn.Linear(np.sum(hidden_size), output_size)
        self.float()

    def forward(self, features):
        # Pass each modality through its respective hidden layers
        x_hidden = [layer(x_modality) for layer, x_modality in zip(self.fc_modality_layers, features)]
        # Concatenate hidden representations
        x_concatenated = torch.cat(x_hidden, dim=-1)
        # Pass concatenated hidden representations through the final fully connected layer
        x_output = self.fc_final(x_concatenated)

        return x_output




class EarlyFusionFeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(EarlyFusionFeedForwardNN, self).__init__()
        hidden_size1 = input_size // 2
        hidden_size2 = hidden_size1 // 2

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.float()  # Set dtype of the model to float

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to output of first layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to output of second layer
        x = self.fc3(x)  # Output of third layer
        return x

