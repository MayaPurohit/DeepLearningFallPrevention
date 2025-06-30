import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, batch_first=True):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first)
        
      
        self.fc = nn.Linear(hidden_size, num_classes)  # Assuming a single output

    def forward(self, x):
        out, _ = self.lstm(x)
        
        out = self.fc(out[:, -1, :])
        return out