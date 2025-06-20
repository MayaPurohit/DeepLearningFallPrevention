

# Maya Purohit

# TestCNN.py
# Develop classes to develop a simple test model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    #extracting the feature 
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(channels, channels, 3, 1, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, 1, padding=1)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.relu = nn.ReLU()
        pass
        
    def forward(self, x):
   
        residual = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
 
        x = self.relu(x)
        return x 
    


class TestCNN(nn.Module):
    def __init__(self, input_channels=6, num_classes=5, num_blocks=2, num_features =  16):
        """
  
        """
        super(TestCNN, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, num_features, 3, 1, padding=4),
            nn.ReLU()
        )

      
        blocks = []
        for i in range(num_blocks):
            block = ResidualBlock(num_features)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.mid_conv = nn.Sequential(
            nn.Conv1d(num_features, num_features, 3, 1, padding=1),
            nn.BatchNorm1d(num_features)
        )

    
        self.fc_layer = nn.Linear(num_features, num_classes)
       
        self._initialize_weights()
        pass
        
    def _initialize_weights(self):

        for m in self.blocks: #initialize the appropriate weights
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        pass
        
    def forward(self, x):
        
        x = self.initial_conv(x)
        residual = x.clone()
       
        x = self.blocks(x)
       
        x = self.mid_conv(x)
        
        x += residual

        x = torch.mean(x, dim=2)
       
        x = self.fc_layer(x)
        return x
    