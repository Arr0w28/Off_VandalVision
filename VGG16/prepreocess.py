import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

class preProcessingTools:

    def __init__(self) -> None:

        self.transform = transforms.Compose(
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )

        self.device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    def ImageProcessing(self,image):
        img_RGB = image.convert('RGB')
        transformed_img = self.transform(img_RGB).unsqueeze(0).to(self.device)
        return transformed_img
