
from pathlib import Path
import random
import shutil

# Folder dataset
RAW_DIR = Path("dataset_raw")
OUT_DIR = Path("dataset_split")

# Pembagian data
TRAIN_RATIO = 0.8
random.seed(42)

CLASSES = ["red box", "black box", "green box"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for class_name in CLASSES:
    source_dir = RAW_DIR / class_name

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Folder tidak ditemukan: {source_dir}"
        )

    images = [
        file for file in source_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if len(images) < 2:
        raise ValueError(
            f"Foto kelas {class_name} kurang dari 2."
        )

    random.shuffle(images)
    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    for split_name, image_list in [
        ("train", train_images),
        ("val", val_images),
    ]:
        destination = OUT_DIR / split_name / class_name
        destination.mkdir(parents=True, exist_ok=True)

        for image_path in image_list:
            shutil.copy2(
                image_path,
                destination / image_path.name
            )

    print(
        f"{class_name}: "
        f"{len(train_images)} train, "
        f"{len(val_images)} validation"
    )

print("\nPembagian dataset selesai!")
print(f"Hasil tersimpan di: {OUT_DIR.resolve()}")