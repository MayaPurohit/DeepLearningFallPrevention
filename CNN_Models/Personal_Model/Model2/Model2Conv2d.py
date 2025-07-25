

# Maya Purohit

# Model.py
# Develop classes to develop a simple test model with self attention to test

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        
        
     
        self.query = nn.Linear(self.embed_size, self.embed_size)
        self.key = nn.Linear(self.embed_size, self.embed_size)
        self.value = nn.Linear(self.embed_size, self.embed_size)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

     
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        #transform into proabilities
        attention_weights = F.softmax(scores, dim=-1)
        
        #get weighted values 
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x, mask=None):
        #make the query, key, and value models
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        
        out, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        return out

  
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





        # blocks = []
        # for i in range(num_blocks):
        #     block = ResidualBlock(num_features*4)
        #     blocks.append(block)

        # self.blocks = nn.Sequential(*blocks)

        self.block3 = ResidualBlock(num_features*4)
        self.attention = SelfAttention(num_features*4)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_layer = nn.Linear(num_features*4, num_features*4)
        self.fc_bn = nn.LayerNorm(num_features*4)
        self.dropout =  nn.Dropout(0.4)
        self.fc_layer2 = nn.Linear(num_features*4, num_classes)
       
        self._initialize_weights()
        pass
        
    def _initialize_weights(self):

        for m in self.modules(): #initialize the appropriate weights
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        pass
        
    def forward(self, x):
        
        x = x.view(x.size(0), 1, self.num_stack*2, 50) 
        x = self.initial_conv(x)

        x = self.block1(x)
       
        x = self.mid_conv(x)
        
        x = self.block2(x)

        x = self.mid_conv2(x)

        x = self.block3(x)



        if self.include_attention == True:
            x = x.view(x.size(0), x.size(1), -1)  # [10, 200, 24]
            x = x.permute(0, 2, 1) 

            x  = self.attention(x)

            x = x.permute(0, 2, 1)               # [10, 200, 24]
            x = x.view(x.size(0), x.size(1), 1, x.size(2))
            
            # x = x.permute(1, 2, 0)


        # x = self.blocks(x)



        x = self.pool(x)
        x = x.squeeze(-1)   
        x = torch.flatten(x, 1)
       
        x = self.fc_layer(x)
        x = self.fc_bn(x)
        x = self.dropout(x)
        x = self.fc_layer2(x)
        return x
    