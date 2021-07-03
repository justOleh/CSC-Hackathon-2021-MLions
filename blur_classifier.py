
import torch
import torch.nn as nn
import torchvision
import cv2 as cv
from torchvision import datasets, models, transforms
from pathlib import Path


transforms.Compose([
             Image.fromarray,  
             transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ]),
 
model_ft = models.resnet18(pretrained=True)
    
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

img = cv.imread('drive/MyDrive/CSC_Hackathon_2021_MLions/IMG_2739.png')
# img = cv.imread('IMG_2749.png')
img_preproc = torch.unsqueeze(data_transforms['val'](img), dim=0)

pred_class = torch.argmax(model_ft(img_preproc.to(device)))

# weights - CSC_Hackathon_2021_MLions/blur_classifier.pth
