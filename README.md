# Knowledge Distillation on CIFAR-100

## ResNet-50 → MobileNetV3-Small

This project implements **Knowledge Distillation (KD)** as introduced in:

> Geoffrey Hinton et al., *Distilling the Knowledge in a Neural Network* (2015)

A high-capacity **ResNet-50 teacher** is trained on CIFAR-100 and used to supervise a smaller **MobileNetV3-Small student** via soft targets. The objective is to evaluate how teacher guidance improves compact model performance.

---

# Overview

Knowledge distillation transfers information from a large model (teacher) to a smaller model (student) by matching softened output distributions.

The student minimizes:

L = α * CE(y, p_s) + (1 - α) * T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )


Where:

* α controls the weight of hard-label supervision
* T is the softmax temperature
* CE is cross-entropy with ground-truth labels
* KL is the distillation loss between teacher and student logits

---

# Environment Setup (Conda + CUDA)

Create and activate environment:

```
conda create -n kd python=3.10 -y
conda activate kd
```

Install PyTorch with CUDA:

```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tqdm pandas matplotlib
```

Verify GPU:

```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

---

# Training Pipeline

## 1) Train Teacher (ResNet-50)

```
python train_teacher.py --out_dir runs/teacher_resnet50 --epochs 40 --batch_size 256 --lr 0.05
```

The teacher is initialized with ImageNet weights and fine-tuned on CIFAR-100.

---

## 2) Train Baseline Student

```
python train_student.py --out_dir runs/student_baseline --epochs 120 --batch_size 256 --lr 0.2
```

This trains MobileNetV3-Small using standard cross-entropy.

---

## 3) Train Student with Knowledge Distillation

```
python distill_student.py --out_dir runs/student_kd --teacher_ckpt runs/teacher_resnet50/best.pt --epochs 120 --alpha 0.5 --T 4
```

The teacher model is frozen during distillation.

---

# Hyperparameters

* α: weight on hard-label cross-entropy
* T: temperature for softmax smoothing
* Epochs: student training duration
* Batch size: default 256
* AMP: mixed precision enabled

---

# Implementation Details

* Dataset: CIFAR-100
* Optimizer: SGD with momentum
* Learning rate schedule: cosine decay
* Mixed precision (AMP) enabled
* Automatic checkpoint saving
* CSV logging for analysis

---

# Why Knowledge Distillation Works

Soft targets from the teacher encode:

* Class similarity structure
* Dark knowledge (probabilities over non-true classes)
* Smoother decision boundaries
* Improved calibration

Hard labels alone do not provide this information.
