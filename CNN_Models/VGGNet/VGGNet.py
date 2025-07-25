

# Maya Purohit
# AlexNet
# Develop classes to develop the AlexNet Architecture Model

import torch
import torch.nn as nn
import torch.nn.functional as F





class VGGNetCNN(nn.Module):
    def __init__(self, input_channels=6, num_classes=5):
        """
  
        """
        super(VGGNetCNN, self).__init__()
        

        self.first_conv = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()   
        )
        

        self.second_conv = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size = 3, stride = 1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride = 2)
            
        )

      
        self.third_conv = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()   
        )
        

        self.fourth_conv = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size = 3, stride = 1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride = 2)
        )

        self.fifth_conv = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size = 3,  padding=1),
            nn.ReLU()
            
        )
        self.sixth_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 3,  padding=1),
            nn.ReLU()
            
        )
        self.seventh_conv = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size = 3,  padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride = 2)
            
        )

        self.eighth_conv = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size = 3,  padding=1),
            nn.ReLU()
            
        )
        self.ninth_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size = 3,  padding=1),
            nn.ReLU()
            
        )
        self.tenth_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size = 3,  padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride = 2)
            
        )

        
        self.eleventh_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size = 3,  padding=1),
            nn.ReLU()
            
        )
        self.twelveth_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size = 3,  padding=1),
            nn.ReLU()
            
        )
        self.thirteenth_conv = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size = 3,  padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride = 2)
            
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256*2, 4096),
            nn.ReLU(),
            nn.Dropout()
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(4096, num_classes),
        )
       
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
        
        x = self.first_conv(x)
       
        x = self.second_conv(x)
       
        x = self.third_conv(x)
         
        x = self.fourth_conv(x)

        x = self.fifth_conv(x)

        x = self.sixth_conv(x)

        x = self.seventh_conv(x)
        x = self.eighth_conv(x)
        x = self.ninth_conv(x)
        x = self.tenth_conv(x)
        x = self.eleventh_conv(x)
        x = self.twelveth_conv(x)
        x = self.thirteenth_conv(x)
       

        x = torch.flatten(x, 1)  

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
      
        return x
    