

# Maya Purohit

# PersModel.py
# Develop classes to develop a simple test model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

    

class PersModelCNN(nn.Module):
    def __init__(self, input_channels=6, num_stacks = 1, num_classes=5, num_features =  16, include_attention = True):
        """
  
        """

        self.include_attention = include_attention
        self.num_stack = num_stacks
        super(PersModelCNN, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

    

        self.mid_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2)
        )

        self.mid_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(64, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_layer2 = nn.Linear(64, 5)

        self._initialize_weights()
        pass
        
    def _initialize_weights(self):

        for m in self.modules(): #initialize the appropriate weights
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        pass
        
    def forward(self, x):

        x = x.view(x.size(0), 1, self.num_stack*2, 50) 
        
        x = self.initial_conv(x)

       
        x = self.mid_conv(x)
        

        x = self.mid_conv2(x)

        x= self.pool(x)
        x = x.squeeze(-1)   
        x = torch.flatten(x, 1)
       
        x = self.fc_layer(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_layer2(x)
        return x
    