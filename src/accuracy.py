import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# -------------------
# Device
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------
# Validation Transform
# -------------------
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------
# Load Validation Dataset
# -------------------
val_dir = "data/images/val"
val_data = datasets.ImageFolder(val_dir, transform=transform_val)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False)

# -------------------
# Load Trained Model
# -------------------
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, len(val_data.classes))

# Load saved weights
model.load_state_dict(torch.load("breed_classifier.pth", map_location=device))
model = model.to(device)
model.eval()

# -------------------
# Predict & Calculate Accuracy
# -------------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Validation Accuracy: {accuracy*100:.2f}%")

# Optional: Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Optional: Detailed Report per Breed
report = classification_report(y_true, y_pred, target_names=val_data.classes)
print("\nClassification Report:")
print(report)
