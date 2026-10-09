
from pathlib import Path
import argparse
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["feature", "partial", "scratch"],
        default="feature"
    )
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    data_dir = Path("dataset_split")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = datasets.ImageFolder(
        data_dir / "train",
        transform=train_transform
    )
    val_dataset = datasets.ImageFolder(
        data_dir / "val",
        transform=val_transform
    )

    if train_dataset.classes != val_dataset.classes:
        raise ValueError(
            "Nama/urutan kelas train dan validation berbeda."
        )

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False
    )

    print("Classes:", train_dataset.classes)
    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    if args.mode == "scratch":
        model = models.resnet18(weights=None)
    else:
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(train_dataset.classes))

    if args.mode == "feature":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        lr = 1e-3

    elif args.mode == "partial":
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
        lr = 1e-4

    else:
        for param in model.parameters():
            param.requires_grad = True
        lr = 1e-3

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)

        model.eval()
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()

        val_acc = correct / len(val_dataset)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc * 100:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    elapsed = time.time() - start_time
    model.load_state_dict(best_weights)

    model_path = output_dir / f"resnet18_{args.mode}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": train_dataset.classes,
        "mode": args.mode
    }, model_path)

    result = (
        f"Mode: {args.mode}\n"
        f"Best validation accuracy: {best_acc * 100:.2f}%\n"
        f"Training time: {elapsed:.2f} seconds\n"
    )
    (output_dir / f"{args.mode}_summary.txt").write_text(
        result, encoding="utf-8"
    )

    plt.figure()
    plt.plot(
        range(1, args.epochs + 1),
        [a * 100 for a in history["val_acc"]],
        marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy (%)")
    plt.title(f"Validation Accuracy - {args.mode}")
    plt.grid(True)
    plt.savefig(
        output_dir / f"{args.mode}_accuracy.png",
        bbox_inches="tight"
    )
    plt.close()

    print("\nTraining selesai.")
    print(result)
    print(f"Model tersimpan: {model_path}")


if __name__ == "__main__":
    main()