

# Maya Purohit

# PersModel.py
# Develop classes to develop a simple test model

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PersModelCNN(nn.Module):
    def __init__(self, input_channels=6, num_classes=5, num_features =  16):
        """
  
        """
        super(PersModelCNN, self).__init__()
        
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels, num_features, 3, 1, padding=4),
            nn.ReLU()
        )

    

        self.mid_conv = nn.Sequential(
            nn.Conv1d(num_features, num_features*2, 3, 1, padding=1),
            nn.BatchNorm1d(num_features*2)
        )

        self.mid_conv2 = nn.Sequential(
            nn.Conv1d(num_features*2, num_features*3, 3, 1, padding=1),
            nn.BatchNorm1d(num_features*3)
        )
    
        self.fc_layer = nn.Linear(num_features*3, num_features*3)
        self.fc_layer2 = nn.Linear(num_features*3, num_classes)
       
        self._initialize_weights()
        pass
        
    def _initialize_weights(self):

        for m in self.modules(): #initialize the appropriate weights
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

       
        x = self.mid_conv(x)
        

        x = self.mid_conv2(x)


        x = torch.mean(x, dim=2)
       
        x = self.fc_layer(x)
        x = self.fc_layer2(x)
        return x
    