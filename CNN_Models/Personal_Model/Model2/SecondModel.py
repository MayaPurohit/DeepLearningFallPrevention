

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
        
        
        # Linear transformation for query, key, and value 
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # Apply mask if provided (useful for masked self-attention in transformers)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to normalize scores, producing attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute the final output as weighted values
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x, mask=None):
        # Generate Q, K, V matrices
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Calculate attention using our scaled dot-product function
        out, _ = scaled_dot_product_attention(Q, K, V, mask)
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
    