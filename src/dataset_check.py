from torchvision import datasets, transforms
from collections import Counter

# Transform (sirf resize aur tensor banane ke liye, augmentation ki zaroorat nahi)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 🟢 Apna dataset load karo
train_data = datasets.ImageFolder("data/images/train", transform=transform)
val_data   = datasets.ImageFolder("data/images/val", transform=transform)

# Classes print karo
print("Classes:", train_data.classes)

# 🟢 Train set counts
train_class_counts = Counter([label for _, label in train_data.samples])
for idx, count in train_class_counts.items():
    print(f"{train_data.classes[idx]} (train): {count} images")

# 🟢 Validation set counts
val_class_counts = Counter([label for _, label in val_data.samples])
for idx, count in val_class_counts.items():
    print(f"{val_data.classes[idx]} (val): {count} images")
