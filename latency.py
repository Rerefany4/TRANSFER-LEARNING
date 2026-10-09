
from pathlib import Path
import time
import statistics

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Pilih model yang akan diuji
MODEL_PATH = Path("results/resnet18_partial.pth")

# Pilih satu gambar validation
DATA_DIR = Path("dataset_split/val")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_paths = [
    p for p in DATA_DIR.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

if not image_paths:
    raise FileNotFoundError(
        "Tidak ditemukan gambar di dataset_split/val"
    )

device = torch.device("cpu")

checkpoint = torch.load(
    MODEL_PATH, map_location=device, weights_only=True
)
classes = checkpoint["classes"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

with Image.open(image_paths[0]) as img:
    image = transform(img.convert("RGB")).unsqueeze(0).to(device)

# Warm-up
with torch.inference_mode():
    for _ in range(5):
        model(image)

# Ukur latency
times = []
with torch.inference_mode():
    for _ in range(50):
        start = time.perf_counter()
        model(image)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

print("Model:", MODEL_PATH.name)
print("Device:", device)
print("Test image:", image_paths[0])
print(f"Mean latency: {statistics.mean(times):.2f} ms")
print(f"Median latency: {statistics.median(times):.2f} ms")
print(f"Approx. throughput: {1000 / statistics.mean(times):.2f} images/s")