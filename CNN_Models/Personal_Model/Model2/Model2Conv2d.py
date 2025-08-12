

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

        self.conv1 = nn.Conv2d(channels, channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, padding=1)

        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

        self.leakyRelu = nn.LeakyReLU(negative_slope=1e-2)

        pass
        
    def forward(self, x):
   
        '''
        Residual Block with skip connection
        '''
        residual = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leakyRelu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
 
        x = self.leakyRelu(x)
        return x 
    

class SecondModelCNN2d(nn.Module):
    def __init__(self, input_channels=6, num_stack = 3, num_classes=5, num_features =  16, num_blocks=2, include_attention = True):
        """
        Model Architecture with 2d Convolutions
        """
        super(SecondModelCNN2d, self).__init__()

        self.include_attention = include_attention
        self.num_stack = num_stack
    
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, num_features, 3, 1, padding=1),
            nn.BatchNorm2d(num_features),
            nn.LeakyReLU(negative_slope=1e-2)

        )

        self.block1 = ResidualBlock(num_features)


        self.mid_conv = nn.Sequential(
            nn.Conv2d(num_features, num_features*2, 3, 1, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.LeakyReLU(negative_slope=1e-2)

        )



        self.block2 = ResidualBlock(num_features*2)
        self.mid_conv2 = nn.Sequential(
            nn.Conv2d(num_features*2, num_features*4, 3, 1, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.LeakyReLU(negative_slope=1e-2),
            nn.MaxPool2d(kernel_size=3, stride = 2)
        )


        self.block3 = ResidualBlock(num_features*4)


        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(num_features*4, num_features*4)
        self.fc_layernorm = nn.LayerNorm(num_features*4)
        self.final_relu = nn.LeakyReLU(negative_slope=1e-2)
        self.dropout =  nn.Dropout(0.4)
        self.fc_layer2 = nn.Linear(num_features*4, num_classes)
       
        self._initialize_weights()
        pass
        
    def _initialize_weights(self):
        '''
        Initialize weights and biases for each type of layer 
        '''
        for m in self.modules(): #initialize the appropriate weights
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        pass
        
    def forward(self, x):
        '''
        Forward pass through the layers 
        '''
        x = x.view(x.size(0), 1, self.num_stack*2, 50) 
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
        x = self.fc_layernorm(x)
        x = self.final_relu(x)
        x = self.dropout(x)
        x = self.fc_layer2(x)
        return x
    