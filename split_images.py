import os
import shutil
import random

SOURCE_DIR = "data/new_images"
DEST_DIR = "data/images"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

def is_image(file):
    return file.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))

for cls in os.listdir(SOURCE_DIR):
    cls_path = os.path.join(SOURCE_DIR, cls)
    if not os.path.isdir(cls_path):
        continue

    images = [f for f in os.listdir(cls_path) if is_image(f)]
    if len(images) == 0:
        print(f"⚠️ No images found for {cls}, skipping")
        continue

    random.shuffle(images)

    n = len(images)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + int(n * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "valid": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        split_dir = os.path.join(DEST_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for f in files:
            src = os.path.join(cls_path, f)
            dst = os.path.join(split_dir, f)

            # prevent overwrite
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

    print(f"✅ {cls} → train:{len(splits['train'])} "
          f"valid:{len(splits['valid'])} test:{len(splits['test'])}")

print("\n🎯 New image classes split completed successfully")
