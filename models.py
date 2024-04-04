import numpy as np
import torch
import torch.nn as nn


class MMFeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        super(MMFeedForwardNN, self).__init__()
        hidden_size = [dim // 2 for dim in input_size]
        self.fc_modality_layers = [nn.Linear(input, hid).to(device) for input,hid in zip(input_size,hidden_size)]
        self.fc_final = nn.Linear(np.sum(hidden_size), output_size)
        self.float()

    def forward(self, features):
        # Pass each modality through its respective hidden layer
        x_hidden = [torch.relu(layer(x_modality))for layer,x_modality in zip(self.fc_modality_layers,features)]
        # Concatenate hidden representations
        x_concatenated = torch.cat(x_hidden, dim=-1)
        # Pass concatenated hidden representations through the final fully connected layer
        x_output = self.fc_final(x_concatenated)

        return x_output
class FeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeedForwardNN, self).__init__()
        hidden_size = input_size // 2
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.float()  # Set dtype of the model to float

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to output of first layer
        x = self.fc2(x)  # Output of second layer
        return x
