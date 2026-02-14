# train_teacher.py
import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models

from utils import (
    set_seed, get_cifar100_loaders, train_one_epoch_ce, evaluate,
    cosine_lr, save_checkpoint, CSVLogger, count_params, measure_inference_ms
)

def make_teacher(pretrained: bool = True) -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 100)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./runs/teacher_resnet50")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_cifar100_loaders(args.data_dir, args.batch_size, args.num_workers)

    model = make_teacher(pretrained=(not args.no_pretrained)).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    logger = CSVLogger(
        os.path.join(args.out_dir, "log.csv"),
        fieldnames=["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc"]
    )

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        lr = cosine_lr(optimizer, epoch - 1, args.epochs, args.lr, min_lr=0.0)

        tr = train_one_epoch_ce(
            model, train_loader, optimizer, scaler, device,
            amp=(args.amp and device.type == "cuda"), grad_clip=args.grad_clip
        )
        va = evaluate(model, val_loader, device)

        row = {"epoch": epoch, "lr": lr, **tr, **va}
        logger.log(row)
        print(row)

        if va["val_acc"] > best_acc:
            best_acc = va["val_acc"]
            save_checkpoint(os.path.join(args.out_dir, "best.pt"), {
                "epoch": epoch,
                "model": model.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
            })

    # quick stats
    p = count_params(model)
    ms = measure_inference_ms(model, device=device, batch_size=256, iters=50) if device.type == "cuda" else float("nan")
    print(f"Teacher params: {p:,} | Inference: {ms:.2f} ms/batch(256) | Best val acc: {best_acc:.4f}")
    logger.close()

if __name__ == "__main__":
    main()
