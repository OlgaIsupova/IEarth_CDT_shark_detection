"""
Lightweight transfer-learning classifier trainer for ImageFolder crops.

Features:
- Train ResNet50 / MobileNetV3-Large / EfficientNet-B0 (torchvision)
- Uses ImageFolder layout: data_dir/train, data_dir/valid, data_dir/test
- Computes class weights automatically and applies to CrossEntropyLoss
- Saves best checkpoint and label map
- Inference helper supports optional reject-threshold: if max prob < threshold -> "background"
- If you already have a "background" class folder, the model will treat it as a regular class.
  If you prefer implicit background detection without a labeled background class, pass
  --reject-threshold 0.5 (default 0.0 = disabled).

Run example:
python /home/DanilOlyaMark/work/SharkDetection/models/train_classifier.py \
  --data-dir /data/Roboflow_shark_species_crops --out-dir ./weights --arch resnet50 \
  --epochs 10 --batch-size 32 --lr 1e-4 --reject-threshold 0.4
"""

import argparse
import json
import os
import re
from pathlib import Path
from pprint import pprint

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def get_model(arch, num_classes, pretrained=True):
    if arch == "resnet50":
        m = models.resnet50(pretrained=pretrained)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(pretrained=pretrained)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif arch == "efficientnet_b0":
        # torchvision >= 0.13
        m = models.efficientnet_b0(pretrained=pretrained)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError("Unknown arch: " + arch)
    return m


def make_loaders(data_dir, batch_size, num_workers=4, img_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.15, 0.15, 0.1, 0.05),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    # load train first so we can reuse its class->idx mapping
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    train_class_to_idx = train_ds.class_to_idx.copy()

    def load_and_remap(split_dir, transform):
        if not os.path.isdir(split_dir):
            return None
        ds = datasets.ImageFolder(split_dir, transform=transform)
        # remap samples to train's class indices; skip unknown classes
        new_samples = []
        skipped = 0
        for path, _ in ds.samples:
            class_name = Path(path).parent.name
            if class_name in train_class_to_idx:
                new_samples.append((path, train_class_to_idx[class_name]))
            else:
                skipped += 1
        if skipped:
            print(
                f"Warning: {skipped} samples in {split_dir} had classes not present in train and were skipped."
            )
        ds.samples = new_samples
        ds.targets = [s[1] for s in new_samples]
        # set classes and mapping to match train
        ds.class_to_idx = train_class_to_idx
        ds.classes = list(
            sorted(train_class_to_idx, key=lambda k: train_class_to_idx[k])
        )
        return ds

    valid_ds = load_and_remap(valid_dir, val_tf)
    test_ds = load_and_remap(test_dir, val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if valid_ds
        else None
    )
    test_loader = (
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        if test_ds
        else None
    )

    # quick sanity print
    print("Class mapping (train):", train_ds.class_to_idx)
    if valid_ds:
        print(
            "Valid samples per class (remapped):",
            {
                c: sum(1 for t in valid_ds.targets if t == idx)
                for c, idx in train_class_to_idx.items()
            },
        )

    return train_ds, valid_ds, test_ds, train_loader, val_loader, test_loader


def compute_class_weights(dataset):
    counts = {}
    for _, y in dataset.samples:
        counts[y] = counts.get(y, 0) + 1
    # order matches dataset.classes
    freq = torch.tensor(
        [counts.get(i, 0) for i in range(len(dataset.classes))], dtype=torch.float
    )
    # avoid zero division
    freq[freq == 0] = 1.0
    weights = 1.0 / freq
    weights = weights / weights.sum() * len(weights)  # normalize roughly
    return weights


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = out.max(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        _, preds = out.max(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def save_checkpoint(state, out_dir, name="best.pth"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    torch.save(state, os.path.join(out_dir, name))


def find_latest_checkpoint(out_dir):
    """
    Return best candidate checkpoint path from out_dir.
    Preference: highest epoch number from epoch_*.pth, then final.pth, then best.pth, else None.
    """
    out = Path(out_dir)
    if not out.exists():
        return None
    epoch_files = []
    for p in out.glob("epoch_*.pth"):
        m = re.search(r"epoch_(\d+)\.pth$", p.name)
        if m:
            epoch_files.append((int(m.group(1)), str(p)))
    if epoch_files:
        epoch_files.sort(key=lambda x: x[0], reverse=True)
        return epoch_files[0][1]
    final_p = out / "final.pth"
    if final_p.exists():
        return str(final_p)
    best_p = out / "best.pth"
    if best_p.exists():
        return str(best_p)
    return None


def predict_single(
    model, img_pil, transform, device, class_names, reject_threshold=0.0
):
    model.eval()
    x = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu()
        top_p, top_i = torch.max(probs, dim=0)
        if reject_threshold > 0.0 and float(top_p) < reject_threshold:
            return "background", float(top_p)
        return class_names[int(top_i)], float(top_p)


def plot_history(history, out_dir):
    """
    history: dict with keys 'train_loss','val_loss','train_acc','val_acc'
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "training_history.png")
    plt.savefig(out_path)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir", required=True, help="ImageFolder-style root (train/valid/test)"
    )
    p.add_argument(
        "--out-dir", required=True, help="where to save checkpoints and labelmap"
    )
    p.add_argument(
        "--arch",
        default="resnet50",
        choices=["resnet50", "mobilenet_v3_large", "efficientnet_b0"],
    )
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--pretrained", action="store_true", help="use pretrained weights (recommended)"
    )
    p.add_argument(
        "--reject-threshold",
        type=float,
        default=0.0,
        help="if >0, inference will return 'background' when max prob < threshold",
    )
    p.add_argument(
        "--resume", default=None, help="path to checkpoint (.pth) to resume from"
    )
    p.add_argument(
        "--continue-training",
        action="store_true",
        help="if set, look in --out-dir for the latest checkpoint and continue training until --epochs total epochs.",
    )
    p.add_argument(
        "--tb-logdir", default=None, help="tensorboard logdir (default: <out-dir>/runs)"
    )
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    train_ds, valid_ds, test_ds, train_loader, val_loader, test_loader = make_loaders(
        args.data_dir, args.batch_size, num_workers=args.workers, img_size=args.img_size
    )

    num_classes = len(train_ds.classes)
    print("Classes:", train_ds.classes)
    print("Num classes:", num_classes)
    if num_classes < 2:
        raise SystemExit("Need at least 2 classes for training")

    model = get_model(args.arch, num_classes, pretrained=args.pretrained)
    model = model.to(device)

    class_weights = compute_class_weights(train_ds).to(device)
    print("Class weights:", class_weights.tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # TensorBoard writer
    tb_logdir = args.tb_logdir if args.tb_logdir else os.path.join(args.out_dir, "runs")
    writer = SummaryWriter(log_dir=tb_logdir)

    # history tracking
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    best_val_acc = 0.0
    out_dir = args.out_dir

    ck_path = None
    if args.resume:
        ck_path = args.resume
    elif args.continue_training:
        ck_path = find_latest_checkpoint(args.out_dir)
        if ck_path:
            print("Auto-detected checkpoint for continuation:", ck_path)
        else:
            print("No checkpoint found in out-dir; starting from scratch.")

    if ck_path and os.path.exists(ck_path):
        print("Loading checkpoint:", ck_path)
        ck = torch.load(ck_path, map_location=device)
        ck_arch = ck.get("arch")
        if ck_arch and ck_arch != args.arch:
            print(f"Warning: checkpoint arch {ck_arch} != requested arch {args.arch}")
        try:
            model.load_state_dict(ck["model_state"])
        except Exception as e:
            print("Warning: could not fully load model state dict:", e)
        if "optimizer_state" in ck:
            try:
                optimizer.load_state_dict(ck["optimizer_state"])
            except Exception as e:
                print("Warning: could not load optimizer state:", e)
        start_epoch = int(ck.get("epoch", 0))
        best_val_acc = (
            float(ck.get("best_val_acc", 0.0)) if "best_val_acc" in ck else 0.0
        )
        if "history" in ck:
            history = ck["history"]
            print(f"Loaded history with {len(history.get('train_loss', []))} epochs")
        ck_classes = ck.get("class_names")
        if ck_classes and ck_classes != train_ds.classes:
            print(
                "Warning: class names in checkpoint differ from current dataset. Proceeding."
            )
    else:
        print("Starting training from scratch.")

    # Determine epoch range so args.epochs is total epochs to reach
    if args.continue_training and start_epoch > 0:
        if start_epoch >= args.epochs:
            print(
                f"Already at epoch {start_epoch} which is >= requested total epochs {args.epochs}. Nothing to do."
            )
            writer.close()
            return
        epoch_range = range(start_epoch + 1, args.epochs + 1)
    else:
        # normal start-from-scratch or explicit resume where user intends to retrain to total epochs
        # If resume was provided explicitly and epoch in checkpoint < args.epochs, continue to args.epochs.
        # If resume provided but checkpoint epoch >= args.epochs, nothing to do.
        if args.resume and start_epoch > 0:
            if start_epoch >= args.epochs:
                print(
                    f"Checkpoint epoch {start_epoch} >= requested total epochs {args.epochs}. Nothing to do."
                )
                writer.close()
                return
            epoch_range = range(start_epoch + 1, args.epochs + 1)
        else:
            epoch_range = range(1, args.epochs + 1)

    for epoch in epoch_range:
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = train_loss, train_acc
        print(
            f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        # update history and tensorboard
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/train", train_acc, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "class_names": train_ds.classes,
                    "arch": args.arch,
                    "best_val_acc": best_val_acc,
                    "history": history,
                },
                out_dir,
                name="best.pth",
            )
            print("Saved best.pth")

        # periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "class_names": train_ds.classes,
                    "arch": args.arch,
                    "best_val_acc": best_val_acc,
                    "history": history,
                },
                out_dir,
                name=f"epoch_{epoch}.pth",
            )

        # always update training history plot
        try:
            plot_history(history, out_dir)
        except Exception as e:
            print("Warning: could not save training plot:", e)

    # final save
    save_checkpoint(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "class_names": train_ds.classes,
            "arch": args.arch,
            "best_val_acc": best_val_acc,
            "history": history,
        },
        out_dir,
        name="final.pth",
    )
    # save labelmap
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "label_map.json"), "w") as f:
        json.dump(
            {
                "classes": train_ds.classes,
                "reject_threshold": args.reject_threshold,
                "history": history,
                "best_val_acc": best_val_acc,
            },
            f,
            indent=2,
        )

    # quick test on test set (if present)
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    print("Training complete. Label map and checkpoints saved to", out_dir)
    print(
        "Inference note: to use reject-based background detection, load label_map.json and model and apply reject_threshold when predicting."
    )


if __name__ == "__main__":
    main()

    # python scripts/roboflow_shark_species/classifier_training.py --data-dir /data/Roboflow_shark_species_dataset_cropped --out-dir /data/Roboflow_shark_species_dataset_cropped/output/resnet50 --arch resnet50 --epochs 20 --batch-size 24 --lr 1e-4 --workers 2 --pretrained --continue-training --tb-logdir /data/Roboflow_shark_species_dataset_cropped/output/resnet50/tensorboard --reject-threshold 0.4
