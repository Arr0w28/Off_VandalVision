import torch
import torch.nn as nn
from torchvision import models

class ImageClassifierCNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg16(pretrained = False)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg(x)
        x = self.classifier(x)
        return x