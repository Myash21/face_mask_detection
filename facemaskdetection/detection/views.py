from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import numpy as np

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model architecture
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Adjust according to your number of classes
model.load_state_dict(torch.load(r'detection\pytorch_model\model.pth', map_location=device))
model = model.to(device)
model.eval()

# Define the mean and std
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

# Define the image transformations
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

def preprocess_image(image_bytes):
    """Preprocess the input image."""
    image = Image.open(io.BytesIO(image_bytes))
    return data_transforms(image).unsqueeze(0)

def index(request):
    return render(request, 'detection/index.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST' and 'file' in request.FILES:
        file = request.FILES['file']
        if file:
            img_bytes = file.read()
            tensor = preprocess_image(img_bytes).to(device)

            with torch.no_grad():
                outputs = model(tensor)
                _, predicted = torch.max(outputs, 1)
                class_idx = predicted.item()

            # Map the class index to a human-readable string
            prediction_text = "Mask Present" if class_idx == 1 else "Mask Absent"

            return render(request, 'detection/index.html', {'prediction': prediction_text})
    return render(request, 'detection/index.html', {'prediction': 'No file uploaded'})
