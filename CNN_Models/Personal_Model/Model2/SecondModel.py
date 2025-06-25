

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
    
  

class SecondModelCNN(nn.Module):
    def __init__(self, input_channels=6, num_classes=5, num_features =  16):
        """
  
        """
        super(SecondModelCNN, self).__init__()
        
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

        self.attention = SelfAttention(num_features*3)

        self.pool = nn.AdaptiveAvgPool1d(1)
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

        x = x.permute(2, 0, 1)  

        x  = self.attention(x)
        
        x = x.permute(1, 2, 0)


        x = self.pool(x)
        x = x.squeeze(-1)   
        x = torch.flatten(x, 1)
       
        x = self.fc_layer(x)
        x = self.fc_layer2(x)
        return x
    