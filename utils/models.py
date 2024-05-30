import numpy as np
import torch
import torch.nn as nn

class OrginalFunctionXOR(nn.Module):
    def __init__(self,setting):
        super(OrginalFunctionXOR, self).__init__()
        self.setting = setting

    def forward(self, X):
        #org_X = self.map_back_to_label(X)
        if 'unique' in self.setting:
            assert self.setting and self.setting[
                -1].isdigit(), 'Last char has to be the number of the modality that should determine the label (starts with 0)'
            unique_modality = int(self.setting[-1])
            org_X = X[unique_modality]
            org_X = org_X.t()
        else:
            org_X = [x.squeeze(dim=0) if x.dim() > 1 else x for x in X]
            org_X = torch.stack(org_X)
        labels = torch.where(torch.sum(org_X, dim=0) == 1, torch.tensor(1), torch.tensor(0))
        logits_1 = torch.where(labels == 1, torch.tensor(1000.0), torch.tensor(-1000.0))  # Large positive value for class 1
        logits_0 = torch.where(labels == 0, torch.tensor(1000.0), torch.tensor(-1000.0))  # Large positive value for class 0
        try:
            logits = torch.stack((logits_0, logits_1), dim=1)
        except:
            logits = torch.stack((logits_0, logits_1), dim=0)
        return logits

    def map_back_to_label(self,features):
        org_values = []
        assert self.setting and self.setting[
            -1].isdigit(), 'Last char has to be the number of the modality that should determine the label (starts with 0)'
        unique_modality = int(self.setting[-1])
        if self.setting == 'synergy':
            for modality in features:
                try:
                    org_value = torch.where(torch.mean(modality, dim=1) > 0.5, torch.tensor(1), torch.tensor(0))
                except:
                    org_value = torch.where(torch.mean(modality, dim=0) > 0.5, torch.tensor(1), torch.tensor(0))
                org_values.append(org_value)

        else:
            modality = None
            if 'unique' in self.setting:
                assert self.setting and self.setting[
                    -1].isdigit(), 'Last char has to be the number of the modality that should determine the label (starts with 0)'
                unique_modality = int(self.setting[-1])
                modality = features[unique_modality]
            elif self.setting == 'redundancy':
                modality = features[0]

            try:
                first_half = modality[:,:len(modality[-1]) // 2]
                second_half = modality[:,len(modality[-1]) // 2:]
                label_first_half = torch.where(torch.mean(first_half, dim=1) > 0.5, torch.tensor(1), torch.tensor(0))
                label_second_half = torch.where(torch.mean(second_half, dim=1) > 0.5, torch.tensor(1), torch.tensor(0))
                org_values = [label_first_half,label_second_half]
            except:
                first_half = modality[ :len(modality) // 2]
                second_half = modality[ len(modality) // 2:]
                label_first_half = torch.where(torch.mean(first_half) > 0.5, torch.tensor(1), torch.tensor(0))
                label_second_half = torch.where(torch.mean(second_half) > 0.5, torch.tensor(1), torch.tensor(0))
                org_values = [label_first_half, label_second_half]

        return org_values

class IntermediateFusionFeedForwardNN(nn.Module):
    def __init__(self, input_size, output_size):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        super(IntermediateFusionFeedForwardNN, self).__init__()
        input_size = [dim[0] for dim in input_size]
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
        input_size = [dim[0] for dim in input_size]
        hidden_size1 = [dim // 2 for dim in input_size]
        hidden_size2 = [dim // 2 for dim in hidden_size1]

        self.fc_modality_layers1 = [nn.Sequential(
            nn.Linear(input_dim, hid_dim),
        ).to(device) for input_dim, hid_dim in zip(input_size, hidden_size1)]
        self.fc_modality_layers2 = [nn.Sequential(
            nn.Linear(input_dim, hid_dim),
        ).to(device) for input_dim, hid_dim in zip(hidden_size1, hidden_size2)]
        self.fc_final = nn.Linear(np.sum(hidden_size2), output_size)
        self.float()


    def forward(self, features):
        # Pass each modality through its respective hidden layers
        x_hidden1 = [torch.relu(layer(x_modality)) for layer, x_modality in zip(self.fc_modality_layers1, features)]
        x_hidden2 = [torch.relu(layer(x_modality)) for layer, x_modality in zip(self.fc_modality_layers2, x_hidden1)]
        # Concatenate hidden representations
        x_concatenated = torch.cat(x_hidden2, dim=-1)
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

