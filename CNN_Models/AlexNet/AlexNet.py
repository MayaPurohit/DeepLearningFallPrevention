

# Maya Purohit
# AlexNet
# Develop classes to develop the AlexNet Architecture Model

import torch
import torch.nn as nn
import torch.nn.functional as F






class AlexNetCNN(nn.Module):
    def __init__(self, input_channels=6, num_classes=5, num_stacks = 2):
        """
  
        """

        self.num_stack = num_stacks
        super(AlexNetCNN, self).__init__()
        
        #Kernel Size = 11, stride = 4
        self.first_conv = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=3, stride=4, padding=2),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride = 2),
            
        )
        

        self.second_conv = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride = 2),
            
        )

        self.third_conv = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size = 3,padding=1),
            nn.ReLU(),
        )

        self.fourth_conv = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size = 3, padding=1),
            nn.ReLU(),
        )

        self.fifth_conv = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size = 3,  padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride = 2),
        )

        self.fc6 = nn.Sequential(
            nn.Linear(256*2, 4096),
            nn.ReLU(),
            nn.Dropout()
        )
        
        self.fc7 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout()
        )

        self.fc8 = nn.Sequential(
            nn.Linear(4096, num_classes),
        )
       
        self._initialize_weights()
        pass
        
    def _initialize_weights(self):

        for m in self.modules(): #initialize the appropriate weights
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


        
        
    def forward(self, x):
        
       
        x = x.view(x.size(0), 1, self.num_stack*2, 50)

        x = self.first_conv(x)
       
        
        x = self.second_conv(x)
       
      
        x = self.third_conv(x)

        
        x = self.fourth_conv(x)

       
        x = self.fifth_conv(x)


       

        x = torch.flatten(x, 1)  

        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
      
        return x
    