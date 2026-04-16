import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, idx


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []
    all_indices = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels, indices in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_indices.extend(indices.tolist())
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "targets": all_targets,
        "preds": all_preds,
        "indices": all_indices,
    }


def build_model(name: str):
    return SimpleCNN() if name == "cnn" else MLP()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    parser.add_argument("--artifact-dir", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="./data/mnist")
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    val_size = int(0.1 * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(IndexedDataset(train_ds), batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(IndexedDataset(val_ds), batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(IndexedDataset(test_ds), batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * labels.size(0)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    test_acc = accuracy_score(test_metrics["targets"], test_metrics["preds"])
    test_macro_f1 = f1_score(test_metrics["targets"], test_metrics["preds"], average="macro")
    cm = confusion_matrix(test_metrics["targets"], test_metrics["preds"])
    report = classification_report(test_metrics["targets"], test_metrics["preds"])

    misclassified = [
        {
            "index": idx,
            "true": true,
            "pred": pred,
        }
        for idx, true, pred in zip(test_metrics["indices"], test_metrics["targets"], test_metrics["preds"])
        if true != pred
    ]

    summary = {
        "device": str(device),
        "model": args.model,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_acc,
        "test_macro_f1": test_macro_f1,
    }

    print(json.dumps(summary, indent=2))

    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), artifact_dir / "best_model.pt")

        with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump({"summary": summary, "history": history}, f, indent=2)

        pd.DataFrame(history).to_csv(artifact_dir / "history.csv", index=False)
        pd.DataFrame(cm).to_csv(artifact_dir / "test_confusion_matrix.csv", index=False)
        (artifact_dir / "test_classification_report.txt").write_text(report, encoding="utf-8")
        pd.DataFrame(misclassified).to_csv(artifact_dir / "misclassified_examples.csv", index=False)

        summary_md = [
            "# PyTorch Run Summary",
            "",
            f"- device: {summary['device']}",
            f"- model: {summary['model']}",
            f"- epochs_ran: {summary['epochs_ran']}",
            f"- best_epoch: {summary['best_epoch']}",
            f"- best_val_accuracy: {summary['best_val_accuracy']:.4f}",
            f"- test_accuracy: {summary['test_accuracy']:.4f}",
            f"- test_macro_f1: {summary['test_macro_f1']:.4f}",
        ]
        (artifact_dir / "run_summary.md").write_text("\n".join(summary_md), encoding="utf-8")
        print(f"Saved artifacts to {artifact_dir}")


if __name__ == "__main__":
    main()
