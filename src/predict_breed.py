import torch
from torchvision import transforms, models
from PIL import Image

# 🟢 Same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🟢 Load model
model = models.resnet18(weights="IMAGENET1K_V1")
num_features = model.fc.in_features

# Change last layer to match breeds
# NOTE: Ye number training ke waqt classes ke hisaab se hoga
import os
from torchvision import datasets
train_data = datasets.ImageFolder("data/images/train", transform=transform)
model.fc = torch.nn.Linear(num_features, len(train_data.classes))

# Load trained weights
model.load_state_dict(torch.load("breed_classifier.pth"))
model.eval()

# 🟢 Class labels
classes = train_data.classes

# 🟢 Function to predict
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    print(f"Prediction: {classes[predicted.item()]}")

# 🟢 Example
predict_image("data/test_images/cow1.jpg")
import torch
from torchvision import transforms, models
from PIL import Image

# 🟢 Same transforms as training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🟢 Load model
model = models.resnet18(weights="IMAGENET1K_V1")
num_features = model.fc.in_features

# Change last layer to match breeds
# NOTE: Ye number training ke waqt classes ke hisaab se hoga
import os
from torchvision import datasets
train_data = datasets.ImageFolder("data/images/train", transform=transform)
model.fc = torch.nn.Linear(num_features, len(train_data.classes))

# Load trained weights
model.load_state_dict(torch.load("breed_classifier.pth"))
model.eval()

# 🟢 Class labels
classes = train_data.classes

# 🟢 Function to predict
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img_t = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
    print(f"Prediction: {classes[predicted.item()]}")

# 🟢 Example
predict_image("data/test_images/cow1.jpg")
