# distill_student.py
import argparse
import os

import torch
import torch.nn as nn
import torchvision.models as models

from utils import (
    set_seed, get_cifar100_loaders, train_one_epoch_kd, evaluate,
    cosine_lr, load_checkpoint, save_checkpoint, CSVLogger, count_params, measure_inference_ms
)

def make_teacher(pretrained: bool = True) -> nn.Module:
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 100)
    return m

def make_student() -> nn.Module:
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 100)
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./runs/student_mnv3s_kd")
    ap.add_argument("--teacher_ckpt", type=str, default="./runs/teacher_resnet50/best.pt")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.5)   # weight on CE (hard labels)
    ap.add_argument("--T", type=float, default=4.0)       # temperature
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--teacher_pretrained_if_missing", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_cifar100_loaders(args.data_dir, args.batch_size, args.num_workers)

    # Teacher
    teacher = make_teacher(pretrained=args.teacher_pretrained_if_missing).to(device)
    if os.path.exists(args.teacher_ckpt):
        ck = load_checkpoint(args.teacher_ckpt, map_location="cpu")
        teacher.load_state_dict(ck["model"], strict=True)
        print(f"Loaded teacher checkpoint: {args.teacher_ckpt} (best_acc={ck.get('best_acc', None)})")
    else:
        print(f"Teacher ckpt not found at {args.teacher_ckpt}. Using ImageNet-pretrained teacher (domain shift).")

    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student
    student = make_student().to(device)

    optimizer = torch.optim.SGD(
        student.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    logger = CSVLogger(
        os.path.join(args.out_dir, "log.csv"),
        fieldnames=["epoch", "lr", "train_loss", "train_acc", "train_ce", "train_kd", "val_loss", "val_acc", "alpha", "T"]
    )

    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        lr = cosine_lr(optimizer, epoch - 1, args.epochs, args.lr, min_lr=0.0)

        tr = train_one_epoch_kd(
            student, teacher, train_loader, optimizer, scaler, device,
            alpha=args.alpha, T=args.T,
            amp=(args.amp and device.type == "cuda"),
            grad_clip=args.grad_clip
        )
        va = evaluate(student, val_loader, device)

        row = {"epoch": epoch, "lr": lr, **tr, **va, "alpha": args.alpha, "T": args.T}
        logger.log(row)
        print(row)

        if va["val_acc"] > best_acc:
            best_acc = va["val_acc"]
            save_checkpoint(os.path.join(args.out_dir, "best.pt"), {
                "epoch": epoch,
                "model": student.state_dict(),
                "best_acc": best_acc,
                "args": vars(args),
            })

    p = count_params(student)
    ms = measure_inference_ms(student, device=device, batch_size=256, iters=50) if device.type == "cuda" else float("nan")
    print(f"KD Student params: {p:,} | Inference: {ms:.2f} ms/batch(256) | Best val acc: {best_acc:.4f}")
    logger.close()

if __name__ == "__main__":
    main()
