# utils.py
import os
import time
import math
import csv
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)


def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cifar100_loaders(
    data_dir: str,
    batch_size: int,
    num_workers: int = 4,
    val_batch_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    if val_batch_size is None:
        val_batch_size = batch_size

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    train_ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=val_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0)
    )
    return train_loader, test_loader


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def save_checkpoint(path: str, state: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location="cpu") -> Dict:
    return torch.load(path, map_location=map_location)


class CSVLogger:
    def __init__(self, path: str, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = list(fieldnames)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._file.flush()

    def log(self, row: Dict):
        # Fill missing keys with None
        out = {k: row.get(k, None) for k in self.fieldnames}
        self._writer.writerow(out)
        self._file.flush()

    def close(self):
        self._file.close()


def cosine_lr(optimizer, epoch, total_epochs, base_lr, min_lr=0.0):
    # Classic cosine schedule
    t = epoch / max(1, total_epochs)
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_meter = 0.0
    acc_meter = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        b = x.size(0)
        loss_meter += loss.item() * b
        acc_meter += accuracy_top1(logits, y) * b
        n += b

    return {
        "val_loss": loss_meter / n,
        "val_acc": acc_meter / n,
    }


def train_one_epoch_ce(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    amp: bool = True,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    model.train()
    loss_meter = 0.0
    acc_meter = 0.0
    n = 0

    it = tqdm(loader, desc="train", leave=False)
    for x, y in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        if scaler is not None and amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        b = x.size(0)
        loss_meter += loss.item() * b
        acc_meter += accuracy_top1(logits, y) * b
        n += b

        it.set_postfix(loss=loss.item(), acc=acc_meter / n)

    return {
        "train_loss": loss_meter / n,
        "train_acc": acc_meter / n,
    }


def train_one_epoch_kd(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    alpha: float,
    T: float,
    amp: bool = True,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    student.train()
    teacher.eval()

    ce_meter = 0.0
    kd_meter = 0.0
    total_meter = 0.0
    acc_meter = 0.0
    n = 0

    it = tqdm(loader, desc="distill", leave=False)
    for x, y in it:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            t_logits = teacher(x)

        with torch.cuda.amp.autocast(enabled=amp):
            s_logits = student(x)

            ce = F.cross_entropy(s_logits, y)

            # KL( softmax(t/T) || softmax(s/T) )
            # PyTorch KLDiv expects input as log-probs and target as probs.
            log_p_s = F.log_softmax(s_logits / T, dim=1)
            p_t = F.softmax(t_logits / T, dim=1)
            kd = F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)

            loss = alpha * ce + (1.0 - alpha) * kd

        if scaler is not None and amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()

        b = x.size(0)
        ce_meter += ce.item() * b
        kd_meter += kd.item() * b
        total_meter += loss.item() * b
        acc_meter += accuracy_top1(s_logits, y) * b
        n += b

        it.set_postfix(loss=loss.item(), acc=acc_meter / n)

    return {
        "train_loss": total_meter / n,
        "train_acc": acc_meter / n,
        "train_ce": ce_meter / n,
        "train_kd": kd_meter / n,
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def measure_inference_ms(model: nn.Module, device: torch.device, batch_size: int = 256, iters: int = 50) -> float:
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32, device=device)
    # warmup
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = model(x)
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) * 1000.0 / iters
