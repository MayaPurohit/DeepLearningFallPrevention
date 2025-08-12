

# Maya Purohit
# ModifiedAlexNet.py
# Develop classes to create smaller version of the AlexNet Architecture Model

import torch
import torch.nn as nn
import torch.nn.functional as F

    
  
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




class ModAlexNetCNN(nn.Module):
    def __init__(self, input_channels=6, num_blocks = 2, include_attention = False, num_classes=5):
        """
        Reduce size of AlexNet to smaller size 
        """
        super(ModAlexNetCNN, self).__init__()
        
        self.include_attention = include_attention

        self.num_blocks = num_blocks
        self.num_classes = num_classes
        #Kernel Size = 11, stride = 4
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride = 2),
            
        )
        

        self.second_conv = nn.Sequential(
            nn.Conv1d(64, 192, kernel_size = 5, stride = 1, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride = 2),
            
        )

        self.third_conv = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size = 3,padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.fourth_conv = nn.Sequential(
            nn.Conv1d(256, 384, kernel_size = 3, padding=1),
            nn.BatchNorm1d(384),
            nn.ReLU(),
        )

        self.fifth_conv = nn.Sequential(
            nn.Conv1d(384, 384, kernel_size = 3,  padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride = 2),
        )

        blocks = []
        for i in range(num_blocks):
            block = ResidualBlock(384)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc6 = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # self.fc7 = nn.Sequential(
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(),
        #     nn.Dropout()
        # )

        self.fc8 = nn.Sequential(
            nn.Linear(256, self.num_classes),
        )
       
        self._initialize_weights()
        pass
        

        # self.attention = SelfAttention(640)

    def _initialize_weights(self):
        '''
        Initialize weights and biases for each type of layer
        '''
        for m in self.modules(): #initialize the appropriate weights
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean = 0.0, std = 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


        
        
    def forward(self, x):
        
        '''
        Forward pass through the layers 
        '''
        x = self.first_conv(x)
       
        
        x = self.second_conv(x)
       
      
        x = self.third_conv(x)

        
        x = self.fourth_conv(x)

       
        x = self.fifth_conv(x)

        x = self.blocks(x)

        # if self.include_attention == True:
        #     x = x.permute(2, 0, 1)  

        #     x  = self.attention(x)
            
        #     x = x.permute(1, 2, 0)
       
        x = self.pool(x)
        x = x.squeeze(-1)   
        x = torch.flatten(x, 1)  

        x = self.fc6(x)
        # x = self.fc7(x)
        x = self.fc8(x)
      
        return x
    

    