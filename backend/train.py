"""Training, validation, and evaluation pipeline for EfficientNet-B0 Deepfake Detection.

Reads stratified CSV manifests from splits/{train,val,test}.csv (or DF-DATASET/splits/)
and saves experimental checkpoints under model/experiments/ so existing
production models (model/deepfake_detector.pth, model/deepfake_trained.pth) are
never overwritten. Compatible with local CPU/GPU and Google Kaggle T4 GPUs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import config
from model_architecture import build_model

DEFAULT_SPLITS_DIR = (
    config.BASE_DIR / "DF-DATASET" / "splits"
    if (config.BASE_DIR / "DF-DATASET" / "splits").exists()
    else config.BASE_DIR / "splits"
)
DEFAULT_EXPERIMENTS_DIR = config.MODEL_DIR / "experiments"
PROTECTED_MODEL_FILES = {
    (config.MODEL_DIR / "deepfake_detector.pth").resolve(),
    (config.MODEL_DIR / "deepfake_trained.pth").resolve(),
}


def seed_everything(seed: int = 42) -> None:
    """Set reproducible random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_transforms() -> transforms.Compose:
    """Return training preprocessing + augmentation pipeline matching EfficientNet-B0."""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.3, 1.4))],
            p=0.25,
        ),
        transforms.RandomGrayscale(p=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
    ])


def get_eval_transforms() -> transforms.Compose:
    """Return deterministic validation/test preprocessing pipeline matching detector.py."""
    return transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD),
    ])


def resolve_dataset_root(dataset_root: str | Path) -> Path:
    """Resolve dataset root whether it points to Cleaned/ directly or its parent folder."""
    root = Path(dataset_root)
    if (root / "Real").is_dir() and (root / "Fake").is_dir():
        return root
    if (root / "Cleaned" / "Real").is_dir() and (root / "Cleaned" / "Fake").is_dir():
        return root / "Cleaned"
    return root


class DeepfakeCSVDataset(Dataset):
    """PyTorch Dataset backed by a split CSV manifest referencing Cleaned/ images."""

    def __init__(
        self,
        csv_path: str | Path,
        dataset_root: str | Path = config.DATASET_DIR,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.dataset_root = resolve_dataset_root(dataset_root)
        self.transform = transform
        self.samples: list[tuple[Path, int, str, str]] = []

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Split manifest not found: {self.csv_path}")

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row["filepath"]
                label = int(row["label"])
                cls_name = row.get("class_name", "Real" if label == 0 else "Fake")
                prefix = row.get("source_prefix", "")
                full_path = self.dataset_root / rel_path
                self.samples.append((full_path, label, cls_name, prefix))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img_path, label, _cls_name, _prefix = self.samples[index]
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
        if self.transform is not None:
            tensor_img = self.transform(rgb_img)
        else:
            tensor_img = transforms.ToTensor()(rgb_img)
        return tensor_img, label


def compute_binary_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1-score (class 1 = Fake), and 2x2 confusion matrix."""
    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)

    tn = int(np.sum((yt == 0) & (yp == 0)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    tp = int(np.sum((yt == 1) & (yp == 1)))

    total = max(1, tn + fp + fn + tp)
    accuracy = (tn + tp) / total
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-12, precision + recall)

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Create a PyTorch DataLoader handling num_workers=0 and multi-worker modes safely."""
    workers = max(0, int(num_workers))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = False
    return DataLoader(dataset, **loader_kwargs)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    total_epochs: int | None = None,
    phase: str = "Train",
) -> dict[str, Any]:
    """Run one training or evaluation epoch with live tqdm progress and running loss."""
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    total_samples = 0
    all_targets: list[int] = []
    all_preds: list[int] = []

    if epoch is not None and total_epochs is not None:
        desc = f"Epoch {epoch}/{total_epochs} [{phase}]"
    else:
        desc = f"[{phase}]"

    pbar = tqdm(
        loader,
        desc=desc,
        total=len(loader),
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (images, labels) in enumerate(pbar, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size
            avg_loss = running_loss / max(1, total_samples)

            preds = torch.argmax(logits, dim=1)
            all_targets.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

            pbar.set_postfix(
                batch=f"{batch_idx}/{len(loader)}",
                loss=f"{avg_loss:.4f}",
            )

    metrics = compute_binary_metrics(all_targets, all_preds)
    metrics["loss"] = running_loss / max(1, total_samples)
    return metrics


def save_checkpoint_safely(
    state: dict[str, Any],
    checkpoint_path: Path,
) -> None:
    """Save checkpoint while refusing to overwrite protected production model files."""
    resolved = checkpoint_path.resolve()
    if resolved in PROTECTED_MODEL_FILES:
        raise PermissionError(f"Refusing to overwrite protected model file: {resolved}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)


def verify_pipeline(
    splits_dir: Path,
    dataset_dir: Path = config.DATASET_DIR,
    batch_size: int = 8,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Phase 4 verification without starting full training."""
    seed_everything(seed)
    device = config.get_device()
    resolved_root = resolve_dataset_root(dataset_dir)

    print(f"[Verify] Loading CSV split manifests from {splits_dir}...", flush=True)
    datasets = {
        "train": DeepfakeCSVDataset(splits_dir / "train.csv", dataset_root=resolved_root, transform=get_train_transforms()),
        "val": DeepfakeCSVDataset(splits_dir / "val.csv", dataset_root=resolved_root, transform=get_eval_transforms()),
        "test": DeepfakeCSVDataset(splits_dir / "test.csv", dataset_root=resolved_root, transform=get_eval_transforms()),
    }

    split_sets: dict[str, set[Path]] = {}
    summary: dict[str, Any] = {
        "device": str(device),
        "dataset_root": str(resolved_root),
        "splits_dir": str(splits_dir),
        "splits": {},
    }

    for name, ds in datasets.items():
        paths = [s[0] for s in ds.samples]
        missing = sum(1 for p in paths if not p.is_file())
        real_cnt = sum(1 for s in ds.samples if s[1] == 0)
        fake_cnt = sum(1 for s in ds.samples if s[1] == 1)
        split_sets[name] = set(paths)

        loader = build_dataloader(ds, batch_size=batch_size, shuffle=(name == "train"), num_workers=0, pin_memory=False)
        batch_imgs, batch_lbls = next(iter(loader))

        summary["splits"][name] = {
            "total": len(ds),
            "real": real_cnt,
            "fake": fake_cnt,
            "missing_files": missing,
            "batch_shape": list(batch_imgs.shape),
            "batch_dtype": str(batch_imgs.dtype),
            "batch_min": round(float(batch_imgs.min().item()), 4),
            "batch_max": round(float(batch_imgs.max().item()), 4),
            "batch_labels": batch_lbls.tolist(),
        }

    summary["overlaps"] = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }

    val_t1, _ = datasets["val"][0]
    val_t2, _ = datasets["val"][0]
    test_t1, _ = datasets["test"][0]
    test_t2, _ = datasets["test"][0]
    train_draws = [datasets["train"][0][0] for _ in range(6)]
    train_varies = any(not torch.equal(train_draws[0], t) for t in train_draws[1:])

    summary["transform_checks"] = {
        "train_transforms": [type(t).__name__ for t in get_train_transforms().transforms],
        "eval_transforms": [type(t).__name__ for t in get_eval_transforms().transforms],
        "val_deterministic": bool(torch.equal(val_t1, val_t2)),
        "val_max_abs_diff": float((val_t1 - val_t2).abs().max().item()),
        "test_deterministic": bool(torch.equal(test_t1, test_t2)),
        "test_max_abs_diff": float((test_t1 - test_t2).abs().max().item()),
        "train_stochastic_active": bool(train_varies),
    }

    model = build_model(pretrained=False, num_classes=2, device=device)
    model.eval()
    sample_loader = build_dataloader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    sample_x, sample_y = next(iter(sample_loader))
    sample_x = sample_x.to(device)
    sample_y = sample_y.to(device)

    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        logits = model(sample_x)
        probs = model.predict_proba(sample_x)
        loss = criterion(logits, sample_y)

    summary["forward_pass"] = {
        "input_shape": list(sample_x.shape),
        "logits_shape": list(logits.shape),
        "probs_shape": list(probs.shape),
        "prob_row_sums": [round(float(v), 6) for v in probs.sum(dim=1).cpu().tolist()],
        "sample_loss": round(float(loss.item()), 6),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake Detection Training Pipeline (EfficientNet-B0)")
    parser.add_argument("--dataset-dir", type=Path, default=config.DATASET_DIR, help="Path to Cleaned/ dataset root")
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR, help="Path to splits/ CSV folder")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR, help="Directory to save checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Run Phase 4 verification without training")
    args = parser.parse_args()

    if args.verify_only:
        report = verify_pipeline(
            args.splits_dir,
            dataset_dir=args.dataset_dir,
            batch_size=min(args.batch_size, 8),
            seed=args.seed,
        )
        print(json.dumps(report, indent=2), flush=True)
        return

    seed_everything(args.seed)
    device = config.get_device()
    resolved_root = resolve_dataset_root(args.dataset_dir)
    num_workers = max(0, int(args.num_workers))
    pin_mem = device.type == "cuda"

    print("=" * 70, flush=True)
    print("[Train] Deepfake Detection Training Pipeline (EfficientNet-B0)", flush=True)
    print(f"  - Compute Device : {device}", flush=True)
    print(f"  - Dataset Root   : {resolved_root}", flush=True)
    print(f"  - Splits Dir     : {args.splits_dir}", flush=True)
    print(f"  - Output Dir     : {args.output_dir}", flush=True)
    print(f"  - Hyperparams    : epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, num_workers={num_workers}", flush=True)
    print("=" * 70, flush=True)

    print(f"[1/4] Loading dataset splits from {args.splits_dir}...", flush=True)
    train_ds = DeepfakeCSVDataset(args.splits_dir / "train.csv", dataset_root=resolved_root, transform=get_train_transforms())
    val_ds = DeepfakeCSVDataset(args.splits_dir / "val.csv", dataset_root=resolved_root, transform=get_eval_transforms())
    test_ds = DeepfakeCSVDataset(args.splits_dir / "test.csv", dataset_root=resolved_root, transform=get_eval_transforms())
    print(
        f"      Loaded splits -> Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}",
        flush=True,
    )

    print(
        f"[2/4] Creating DataLoaders (batch_size={args.batch_size}, num_workers={num_workers}, pin_memory={pin_mem})...",
        flush=True,
    )
    train_loader = build_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem
    )
    val_loader = build_dataloader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )
    test_loader = build_dataloader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem
    )
    print(
        f"      DataLoaders ready -> Train batches: {len(train_loader):,} | Val batches: {len(val_loader):,} | Test batches: {len(test_loader):,}",
        flush=True,
    )

    use_pretrained = not args.no_pretrained
    print(
        f"[3/4] Loading EfficientNet-B0 model (ImageNet pretrained={use_pretrained}) onto '{device}'...",
        flush=True,
    )
    model = build_model(pretrained=use_pretrained, num_classes=2, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("      Model, CrossEntropyLoss, and AdamW optimizer initialized.", flush=True)

    best_val_f1 = -1.0
    best_ckpt = args.output_dir / "efficientnet_b0_best.pth"
    last_ckpt = args.output_dir / "efficientnet_b0_last.pth"

    print(f"[4/4] Starting training loop for {args.epochs} epoch(s)...", flush=True)
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---", flush=True)
        t0 = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=args.epochs,
            phase="Train",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            epoch=epoch,
            total_epochs=args.epochs,
            phase="Val",
        )
        elapsed = time.time() - t0

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }
        save_checkpoint_safely(state, last_ckpt)

        is_new_best = val_metrics["f1"] > best_val_f1
        if is_new_best:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint_safely(state, best_ckpt)

        print(
            f"Epoch {epoch}/{args.epochs} Summary ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"P: {val_metrics['precision']:.4f} R: {val_metrics['recall']:.4f} F1: {val_metrics['f1']:.4f}"
            + (f" [NEW BEST -> {best_ckpt.name}]" if is_new_best else ""),
            flush=True,
        )

    print("\n[Test] Running final evaluation on test split...", flush=True)
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        epoch=args.epochs,
        total_epochs=args.epochs,
        phase="Test",
    )
    print(f"[Test] Final Metrics: {json.dumps(test_metrics)}", flush=True)


if __name__ == "__main__":
    main()
