import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 🟢 Data Augmentation for Training
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         
                         [0.229, 0.224, 0.225])
])

# 🟢 Validation Data Transform (no augmentation, only resize + normalize)
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 🟢 Dataset Load
train_data = datasets.ImageFolder("data/images/train", transform=transform_train)
val_data   = datasets.ImageFolder("data/images/val", transform=transform_val)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
val_loader   = torch.utils.data.DataLoader(val_data, batch_size=16, shuffle=False)

# 🟢 Model (ResNet18 pretrained)
from torchvision.models import resnet18, ResNet18_Weights
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(train_data.classes))  # Output = breeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 🟢 Loss & Optimizer (smaller LR)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 🟢 Training Loop
EPOCHS = 20   # zyada epochs for better accuracy
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)

    # Validation after each epoch
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")

# 🟢 Save Trained Model
torch.save(model.state_dict(), "breed_classifier.pth")
print("✅ Model trained and saved as breed_classifier.pth")
