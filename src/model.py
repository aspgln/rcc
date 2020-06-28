import torch
import torch.nn as nn
from torchvision import models


class MRNet(nn.Module):
    '''
    Original MRNet architecture
    @input: s x 3 x 256 x 256
    @output: 1 x 2
    '''
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
#         self.dropout = nn.Dropout2d(p=0.5)
        self.classifier = nn.Linear(256, 1)
    
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(flattened_features)
        return output


class MRNet2(nn.Module):
    '''
    Modified classifier
    @input: s x 3 x 256 x 256
    @output: 1 x 2
    '''
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1),
        )    
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(flattened_features)
        return output


    
class MRNetBN(nn.Module):
    '''
    MRNet with batch normalization
    @input: s x 3 x 256 x 256
    @output: 1 x 2
    '''
    '''
    https://nbviewer.jupyter.org/github/KushajveerSingh/Deep-Learning-Notebooks/blob/master/Blog%20Posts%20Notebooks/Training%20AlexNet%20with%20tips%20and%20checks%20on%20how%20to%20train%20CNNs/%281%29.ipynb
    '''
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)
        
        
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(flattened_features)

        return output
        
class MRResNet(nn.Module):
    '''
    MRNet with ResNet architecture
    @input: s x 3 x 256 x 256
    @output: 1 x 2
    '''
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.resnet18(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.classifier = nn.Linear(512, 1)
    
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        mod = nn.Sequential(*list(self.pretrained_model.children())[:-1])
        pooled_features = mod(x)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(flattened_features)
        return output
    
    
class MRNetScratch(nn.Module):
    '''
    MRNet trained from scratch
    @input: s x 3 x 256 x 256
    @output: 1 x 2
    '''
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=False)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.classifier = nn.Linear(256, 1)
    
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(flattened_features)
        return output
        
    

class TDNet(nn.Module):
    '''
    A 2-D network architecture.
    @input: s x 3 x 256 x 256
    @output: s x 2
    '''
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.classifier = nn.Linear(256, 1)
    
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        # this is the only different with the MRNet architecture
        # flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(pooled_features)
        
        return output
    
       
class AlexNet(nn.Module):
    '''
    https://nbviewer.jupyter.org/github/KushajveerSingh/Deep-Learning-Notebooks/blob/master/Blog%20Posts%20Notebooks/Training%20AlexNet%20with%20tips%20and%20checks%20on%20how%20to%20train%20CNNs/%281%29.ipynb
    '''
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)
        
        
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(flattened_features)

        return output

