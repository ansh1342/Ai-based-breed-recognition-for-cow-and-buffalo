# make_breed_folders.py
# Usage: python make_breed_folders.py
# Copies random images from data/images/train/<breed> to static/breeds/<breed> (lowercased)
# Creates static/breeds folders if missing.

import os
import shutil
import random

# ---------- configuration ----------
NUM_IMAGES = 5   # number of images to copy per breed (change if needed)

# Make paths absolute relative to this script location (safer)
BASE = os.path.dirname(os.path.abspath(__file__))
SOURCE_TRAIN = os.path.join(BASE, "data", "images", "train")
DEST_ROOT = os.path.join(BASE, "static", "breeds")
# -----------------------------------

def safe_breed_name(b):
    # normalize breed folder name for destination
    return b.replace(" ", "_").lower()

def main():
    if not os.path.exists(SOURCE_TRAIN):
        print(f"❌ Source train folder not found: {SOURCE_TRAIN}")
        print("Make sure your dataset is at data/images/train")
        return

    os.makedirs(DEST_ROOT, exist_ok=True)
    breeds = [d for d in os.listdir(SOURCE_TRAIN) if os.path.isdir(os.path.join(SOURCE_TRAIN, d))]
    if not breeds:
        print("⚠️  No breed folders found in:", SOURCE_TRAIN)
        return

    total_copied = 0
    for breed in sorted(breeds):
        src_folder = os.path.join(SOURCE_TRAIN, breed)
        dest_folder = os.path.join(DEST_ROOT, safe_breed_name(breed))
        os.makedirs(dest_folder, exist_ok=True)

        all_imgs = [f for f in os.listdir(src_folder)
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))]
        if not all_imgs:
            print(f"⚠️  No images found for breed '{breed}' (skipping).")
            continue

        sample = random.sample(all_imgs, min(NUM_IMAGES, len(all_imgs)))
        copied = 0
        for img in sample:
            src = os.path.join(src_folder, img)
            dst = os.path.join(dest_folder, img)
            try:
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                copied += 1
            except Exception as e:
                print(f"   Error copying {src} -> {dst}: {e}")

        total_copied += copied
        print(f"✅ {breed} -> {dest_folder} : copied {copied} image(s)")

    print("\nDone. Total images copied:", total_copied)
    print("Destination root:", DEST_ROOT)
    print("Now your app can serve images from static/breeds/<breed>/")

if __name__ == "__main__":
    main()
