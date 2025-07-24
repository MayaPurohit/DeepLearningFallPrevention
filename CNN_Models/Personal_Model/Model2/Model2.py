

# Maya Purohit

# Model.py
# Develop classes to develop a simple test model with self attention to test

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch.nn as nn

  
class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    #extracting the feature 
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(channels, channels, 3, 1, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, 1, padding=1)

        self.bn1 = nn.BatchNorm1d(channels)
        self.bn2 = nn.BatchNorm1d(channels)

        self.leakyRelu = nn.LeakyReLU(negative_slope=1e-2)

        pass
        
    def forward(self, x):
   
        residual = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyRelu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
 
        x = self.leakyRelu(x)
        return x 
    

class SecondModelCNN(nn.Module):
    def __init__(self, input_channels=6, num_stack = 3, num_classes=5, num_features =  16, num_blocks=2, include_attention = True):
        """
  
        """
        super(SecondModelCNN, self).__init__()

        self.include_attention = include_attention
        self.num_stack = num_stack
    
        
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_channels*self.num_stack, num_features, 3, 1, padding=1),
            nn.BatchNorm1d(num_features),
            nn.LeakyReLU(negative_slope=1e-2)

        )

        self.block1 = ResidualBlock(num_features)


        self.mid_conv = nn.Sequential(
            nn.Conv1d(num_features, num_features*2, 3, 1, padding=1),
            nn.BatchNorm1d(num_features*2),
            nn.LeakyReLU(negative_slope=1e-2)

        )

        self.block2 = ResidualBlock(num_features*2)

        self.mid_conv2 = nn.Sequential(
            nn.Conv1d(num_features*2, num_features*4, 3, 1, padding=1),
            nn.BatchNorm1d(num_features*4),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.MaxPool1d(kernel_size=3, stride = 2)
        )

        self.block3 = ResidualBlock(num_features*4)




        # blocks = []
        # for i in range(num_blocks):
        #     block = ResidualBlock(num_features*4)
        #     blocks.append(block)

        # self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc_layer = nn.Linear(num_features*4, num_features*4)
        self.fc_bn = nn.LayerNorm(num_features*4)
        self.dropout =  nn.Dropout1d(0.4)
        self.fc_layer2 = nn.Linear(num_features*4, num_classes)
       
        self._initialize_weights()
        pass
        
    def _initialize_weights(self):

        for m in self.modules(): #initialize the appropriate weights
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        pass
        
    def forward(self, x):
        
        x = self.initial_conv(x)

        x = self.block1(x)
       
        x = self.mid_conv(x)
        
        x = self.block2(x)

        x = self.mid_conv2(x)

        x = self.block3(x)







        x = self.pool(x)
        x = x.squeeze(-1)   
        x = torch.flatten(x, 1)
       
        x = self.fc_layer(x)
        x = self.fc_bn(x)
        x = self.dropout(x)
        x = self.fc_layer2(x)
        return x
    