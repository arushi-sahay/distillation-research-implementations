# Paper 01 — Distilling the Knowledge in a Neural Network

**Authors:** Geoffrey Hinton, Oriol Vinyals, and Jeff Dean  
**Year:** 2015  
**Paper:** https://arxiv.org/abs/1503.02531

## Scope

This implementation investigates the central Hinton-style knowledge-distillation idea in a modern image-classification setting using Imagenette v2 (160px). It is an independent implementation of the method, not a reproduction of every experiment in the original paper.

## Dataset

- **Dataset:** Imagenette v2, 160px
- **Source archive:** https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
- **Classes:** ten ImageNet-derived categories
- **Project training split:** 8,522 images (90% stratified subset of the source training split)
- **Internal validation split:** 947 images (10% stratified subset of the source training split)
- **Held-out evaluation split:** 3,925 images (the complete source `val` folder; reserved for final evaluation)
- **Split seed:** 42

The source `val` folder is called the held-out test/evaluation split in this project. It is not used for model selection or tuning.

### Preprocessing

Training uses `RandomResizedCrop(160, scale=(0.75, 1.0), ratio=(0.75, 1.333))`, followed by `RandomHorizontalFlip(p=0.5)`, tensor conversion, and ImageNet mean/std normalisation.

Internal validation and held-out evaluation use `Resize(160)`, `CenterCrop(160)`, tensor conversion, and the same normalisation. The centre-crop choice was made after visually comparing centre cropping with aspect-ratio-preserving letterboxing on representative dataset images.

Dataset images, downloaded archives, checkpoints, and result images are not committed to this repository. The dataset is downloaded at runtime; the rights and terms for the underlying ImageNet photographs remain separate from the fastai repository's software licence.

## Model comparison

All models use TorchVision implementations, are trained from scratch with `weights=None`, and have a 512-to-10 classifier head.

| Stage | Architecture | Supervision / objective |
|---|---|---|
| `teacher` | ResNet-34 | Ground-truth labels with cross-entropy |
| `student_hard` | ResNet-18 | Ground-truth labels with cross-entropy (standard supervised baseline) |
| `student_kd_soft` | ResNet-18 | Teacher soft targets with temperature-scaled KL divergence only |
| `student_kd_combined` | ResNet-18 | Weighted KD loss plus ground-truth cross-entropy |

In this project, **hard labels** means the dataset's ground-truth class labels. It does not mean teacher-generated hard pseudo-labels.

The three ResNet-18 students begin from identical initial parameters and buffers, stored independently. The student training protocols are intended to be matched, including data order and augmentation randomness as far as practical.

## Current experiment configuration

The current configuration records these starting values:

- Optimiser: SGD
- Learning rate: 0.0125
- Momentum: 0.9
- Weight decay: 0.0001
- Epoch target: 90 per stage
- Scheduler: StepLR, step size 30, gamma 0.1
- KD temperature: 4.0
- Combined objective: 0.9 × KD loss + 0.1 × hard-label cross-entropy

These are the initial experiment settings, not empirically tuned optima. Mixed-precision mode and the best-checkpoint selection metric remain to be finalised before training.

## Implementation status

Verified so far:

- Dataset download/extraction, class mapping, stratified split, DataLoaders, batch smoke test, and transformed-image preview.
- ResNet-34 and ResNet-18 construction, classifier dimensions, parameter counts, and forward output shape.
- Identical student initialisation with independent parameter/buffer storage.
- Hard-label, soft-target KD, and combined loss functions, including synthetic loss and gradient-flow checks.
- RNG capture/restoration and atomic checkpoint serialization.
- Checkpoint construction, compatibility validation, and restoration tests, including optimiser, scheduler, epoch, CPU/CUDA RNG, and DataLoader generator state.

**No model training has started yet.** Checkpoint tests used disposable temporary files; no trained-model checkpoint has been produced.

## Notebook

The runnable implementation notebook will be placed at:

`imagenette_distillation.ipynb`

Keep notebook outputs and runtime-specific metadata cleared before committing. Keep downloaded data, checkpoints, and experiment result images outside version control.

## Remaining work before training

- Finalise AMP FP16 with GradScaler on CUDA versus full FP32.
- Finalise the best-checkpoint selection metric (validation accuracy has been proposed; validation loss will also be recorded).
- Implement and test the training and validation routines on a small number of batches.
- Then train the teacher, followed by the three students.

## Reproducibility and recovery

Each stage has its own checkpoint directory. `latest.pt` is intended for epoch-boundary resumption and `best_model.pt` for evaluation. The latest checkpoint preserves model, optimiser, scheduler/scaler when used, epoch progress, best metric, training history, configuration, RNG states, and DataLoader generator states. Resuming after a completed epoch continues at the next epoch; unfinished minibatches within an interrupted epoch are not currently recovered.
