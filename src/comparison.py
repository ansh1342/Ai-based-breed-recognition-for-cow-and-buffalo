import os
from collections import Counter
from torchvision import datasets, transforms

# -------------------
# PATH SET KARNA HAI
# -------------------
train_dir = "data/images/train"
val_dir = "data/images/val"

# Normal transform (sirf check ke liye, resize etc. zaroori nahi)
transform = transforms.Compose([transforms.ToTensor()])

# PyTorch dataset
train_data = datasets.ImageFolder(root=train_dir, transform=transform)
val_data = datasets.ImageFolder(root=val_dir, transform=transform)

# PyTorch counts
train_counts = Counter([label for _, label in train_data.samples])
val_counts = Counter([label for _, label in val_data.samples])

print("✅ Comparing Dataset Counts (Folder vs PyTorch)\n")

for idx, breed in enumerate(train_data.classes):
    # Folder count (train)
    train_folder = os.path.join(train_dir, breed)
    train_files = [f for f in os.listdir(train_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))]
    train_folder_count = len(train_files)

    # PyTorch count (train)
    train_pytorch_count = train_counts[idx]

    # Folder count (val)
    val_folder = os.path.join(val_dir, breed)
    val_files = [f for f in os.listdir(val_folder) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'))]
    val_folder_count = len(val_files)

    # PyTorch count (val)
    val_pytorch_count = val_counts[idx]

    print(f"{breed}:")
    print(f"   Train -> Folder: {train_folder_count}, PyTorch: {train_pytorch_count}")
    print(f"   Val   -> Folder: {val_folder_count}, PyTorch: {val_pytorch_count}\n")
