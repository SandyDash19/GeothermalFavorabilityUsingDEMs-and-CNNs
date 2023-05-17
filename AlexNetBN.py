import torch
from torch import nn

class AlexNetBN(nn.Module):
    def __init__(self, lr=0.1, num_classes=1):
        super().__init__()       
        self.net = nn.Sequential(
            # define in_channels = 1 because I am passing grayscale images. 
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=2), 
            #nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, stride=1, padding=1), 
            #nn.BatchNorm2d(384),
            nn.ReLU(),

            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1), 
            #nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            #nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            #nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.net(x)        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    
    

