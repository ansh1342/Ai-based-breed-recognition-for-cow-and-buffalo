import os
import random
import shutil

# Input folder jaha abhi saare breeds ke alag-alag folders hain
INPUT_DIR = "data/all_images"   # 👈 yaha abhi saara raw data daal
OUTPUT_DIR = "data/images"      # 👈 yaha train/val folder banenge

TRAIN_RATIO = 0.8  # 80% train, 20% val

# Seed fix kar diya taaki har baar random same result de
random.seed(42)

# Har breed folder ke liye loop
for breed in os.listdir(INPUT_DIR):
    breed_path = os.path.join(INPUT_DIR, breed)
    if not os.path.isdir(breed_path):
        continue

    # Images list
    images = [f for f in os.listdir(breed_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    # Split
    split_idx = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Output path
    train_out = os.path.join(OUTPUT_DIR, "train", breed)
    val_out = os.path.join(OUTPUT_DIR, "val", breed)
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy(os.path.join(breed_path, img), os.path.join(train_out, img))
    for img in val_images:
        shutil.copy(os.path.join(breed_path, img), os.path.join(val_out, img))

    print(f"{breed}: {len(train_images)} train, {len(val_images)} val")

print("✅ Splitting done!")
