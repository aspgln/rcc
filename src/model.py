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
        self.dropout = nn.Dropout2d(p=0.5)
        self.classifier = nn.Linear(256, 2)
    
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
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
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.classifier = nn.Linear(256, 2)
    
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
        self.classifier = nn.Linear(256, 2)
    
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        # this is the only different with the MRNet architecture
        # flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(pooled_features)
        return output
    
       
class MRNetBN(nn.Module):
    '''
    MRNet with batch normalization
    @input: s x 3 x 256 x 256
    @output: 1 x 2
    '''
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(pretrained=True)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout2d(p=0.5)
        self.classifier = nn.Linear(256, 2)
    
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
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
        self.classifier = nn.Linear(256, 2)
    
    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifier(flattened_features)
        return output