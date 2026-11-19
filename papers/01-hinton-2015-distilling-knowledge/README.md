# Paper 01 — Distilling the Knowledge in a Neural Network

**Authors:** Geoffrey Hinton, Oriol Vinyals, and Jeff Dean  
**Year:** 2015  
**Paper:** [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

> **Status:** The initial four-stage training run is complete. The reported numbers below are best internal-validation results from one run per model. The held-out test split has not yet been evaluated. The experimental design is being reviewed before further training.

## 1. What this implementation investigates

This project implements a Hinton-style teacher-student distillation setup on Imagenette v2 at 160-pixel resolution. It compares a supervised teacher, a supervised student baseline, and two students trained with distillation objectives.

This is an independent implementation in a modern image-classification setting. It is **not** a reproduction of every experiment in the 2015 paper, and the current results do not establish that distillation improves performance in this setup.

## 2. Dataset and split design

- **Dataset:** Imagenette v2, 160px variant.
- **Source archive:** [`imagenette2-160.tgz`](https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz).
- **Classes:** 10 ImageNet-derived classes: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, and parachute.
- **Source training folder:** 9,469 images.
- **Training subset:** 8,522 images.
- **Internal validation subset:** 947 images.
- **Held-out test subset:** 3,925 images (the complete source `val` folder).
- **Split procedure:** 90/10 stratified split of the source training folder, using seed 42.

The source dataset calls the 3,925-image folder `val`. In this project it is reserved as a final held-out **test** split. It was not used for gradient updates, checkpoint selection, or the reported results in this README. It remains unevaluated at the time of writing.

### Preprocessing

Training transforms:

1. `RandomResizedCrop(160, scale=(0.75, 1.0), ratio=(0.75, 1.333))`.
2. `RandomHorizontalFlip(p=0.5)`.
3. Tensor conversion.
4. ImageNet mean/std normalisation: mean `(0.485, 0.456, 0.406)`, standard deviation `(0.229, 0.224, 0.225)`.

Internal validation and held-out evaluation transforms:

1. `Resize(160)`.
2. `CenterCrop(160)`.
3. Tensor conversion and the same normalisation.

The implementation creates separate training and evaluation views of the source training images so that training augmentation is not applied to internal validation images.

## 3. Models and training objectives

All models use TorchVision ResNet architectures with a 10-class output head. All were initialized from scratch with `weights=None`; no pretrained weights were used in this first run.

| Stage | Architecture | Parameters | Training objective |
|---|---|---:|---|
| `teacher` | ResNet-34 | 21,289,802 | Ground-truth cross-entropy |
| `student_hard` | ResNet-18 | 11,181,642 | Ground-truth cross-entropy |
| `student_kd_soft` | ResNet-18 | 11,181,642 | Temperature-scaled teacher-to-student KL divergence |
| `student_kd_combined` | ResNet-18 | 11,181,642 | 0.9 × KD loss + 0.1 × ground-truth cross-entropy |

The three student models started from identical initial parameters and BatchNorm buffers, stored as independent model states. This controls the initialization across the student-objective comparison.

In this repository, **hard labels** means the dataset's ground-truth class labels. It does not mean teacher-generated hard pseudo-labels.

### Distillation loss

The soft-target objective compares the teacher and student distributions at temperature \(T=4\):

\[
L_{KD} = T^2\, KL\left(\operatorname{softmax}(z_t/T)\;\middle\|\;\operatorname{softmax}(z_s/T)\right)
\]

The implementation uses `KLDivLoss` with `reduction="batchmean"` and multiplies the result by \(T^2\). The teacher is in evaluation mode, its parameters have gradients disabled, and its forward pass is wrapped in `torch.no_grad()` during student training.

The combined objective is:

\[
L_{combined}=0.9L_{KD}+0.1L_{CE}
\]

Temperature 4.0 was chosen as a paper-informed starting value, not as an Imagenette-tuned optimum. The same temperature and loss weights were used in both distillation stages for a controlled comparison.

## 4. Training configuration

The initial run used the following common settings:

| Setting | Value |
|---|---:|
| Epoch target | 90 per model |
| Batch size | 32 |
| Optimizer | SGD |
| Initial learning rate | 0.0125 |
| Momentum | 0.9 |
| Weight decay | 0.0001 |
| Scheduler | StepLR |
| Scheduler step size | 30 epochs |
| Scheduler gamma | 0.1 |
| Random seed | 42 |
| KD temperature | 4.0 |
| Combined-loss KD weight | 0.9 |
| Combined-loss CE weight | 0.1 |
| Precision on CUDA | AMP FP16 with gradient scaling |
| CPU fallback | FP32 |

These are the first-run settings, not a tuned configuration. Each model was evaluated on the full 947-image internal validation subset after every epoch. The best checkpoint was selected by maximum internal-validation accuracy; validation loss was also recorded.

## 5. Results from the first run

**Best internal-validation accuracy (single run per model):**

| Model | Best epoch | Best internal-validation accuracy |
|---|---:|---:|
| ResNet-34 teacher | 54 | 84.6885% |
| ResNet-18 student, hard labels | 54 | **85.8501%** |
| ResNet-18 student, soft KD only | 53 | 85.3221% |
| ResNet-18 student, combined KD + CE | 86 | 84.8997% |

The first run did not show an accuracy gain from distillation over the supervised student baseline:

- Soft-only KD was approximately **0.53 percentage points below** the hard-label student.
- Combined KD was approximately **0.95 percentage points below** the hard-label student.
- The ResNet-34 teacher's best internal-validation accuracy was approximately **1.16 percentage points below** the hard-label student.

These are descriptive differences on the internal validation set, not statistically established improvements or regressions.

### What these results do—and do not—show

The result is a useful diagnostic: in this particular from-scratch setup, the teacher did not demonstrate a clear predictive-performance advantage over a student trained directly on the ground-truth labels. Consequently, the experiment did not establish a teacher-quality surplus that would make a distillation gain likely.

The teacher also showed a substantial train/validation gap. At epoch 90, its training accuracy was approximately 99.72%, while its best internal-validation accuracy across training was 84.69% (the epoch-90 validation accuracy was approximately 84.37%). This is consistent with overfitting and raises a concern about how useful its predictions are as targets for the student.

However, top-1 accuracy alone does **not** tell us whether the teacher's full probability distributions contain useful class-relationship information. We have not yet analysed teacher calibration, per-class confusions, top-k behaviour, or the similarity and information content of its soft targets. It would therefore be too strong to conclude that the teacher had nothing useful to teach, or that knowledge distillation is ineffective in general.

## 6. Limitations of this first experiment

1. **Teacher quality was not established.** Both teacher and students were trained from scratch on a relatively small dataset. The larger ResNet-34 did not achieve higher internal-validation accuracy than the ResNet-18 supervised baseline.
2. **Overfitting is a concern.** The teacher's near-saturated training accuracy and much lower validation accuracy indicate a substantial generalization gap.
3. **One seed per model.** The results do not measure run-to-run variation. Small accuracy differences on 947 validation images should not be overinterpreted.
4. **Validation-based checkpoint selection.** Each reported score is the best result selected using the same internal validation set. It is therefore a model-selection result, not an unbiased final generalization estimate.
5. **Held-out test evaluation is pending.** The 3,925-image source `val` folder was reserved and has not been used to evaluate the checkpoints.
6. **No distribution-quality analysis yet.** Calibration, confidence, class-wise confusions, and teacher-student output relationships have not been examined.
7. **The implementation is still notebook-centred.** The current notebook contains the experiment code, but the planned modular Python package and one-command CLI have not been implemented.

The current experiment should be described as a completed **initial training run**, not as a completed reproduction or a demonstrated improvement from distillation.

## 7. What is planned next

The next phase is a redesign, not an attempt to make the current numbers look better.

### A. Preserve and evaluate this baseline

- Keep the four existing best and latest checkpoints as the record of the initial run.
- Evaluate the best checkpoints on the untouched 3,925-image test split after fixing the evaluation protocol.
- Report test results separately from the validation results above.
- Inspect confusion matrices, per-class accuracy, top-k accuracy, and calibration-related metrics where appropriate.

### B. Improve and validate the teacher before spending compute on new KD runs

- Investigate stronger regularisation and augmentation for a from-scratch teacher.
- Consider a stronger teacher or an ensemble, but verify that it generalizes on the internal validation split before using it for new student experiments.
- Treat ImageNet-pretrained weights cautiously: Imagenette is derived from ImageNet, so external pretraining must be disclosed and potential image overlap/data contamination must be assessed before claiming a clean held-out evaluation.
- Do not use the held-out test split to choose the teacher, temperature, or training configuration.

### C. Make comparisons more reliable

- Run multiple independent seeds (target: at least three) for the redesigned comparisons.
- Report mean and standard deviation, and use paired comparisons where the evaluation design supports them.
- Keep the hard-label baseline matched to the distillation students in architecture, initialization protocol, data split, and training budget.
- Treat temperature and loss weights as experimental variables to tune on validation data, not as universally optimal constants.

### D. Refactor the implementation into modular Python files

The planned layout separates concerns so that architecture, training, configuration, data handling, losses, and evaluation can be inspected independently. The Colab notebook will remain as a convenient execution interface, while the underlying implementation moves into importable Python modules and a small command-line entry point.

A target structure is:

```text
papers/01-hinton-2015-distilling-knowledge/
├── README.md
├── config/
│   └── experiment.yaml
├── src/
│   └── distillation_paper01/
│       ├── models.py
│       ├── data.py
│       ├── losses.py
│       ├── config.py
│       ├── engine.py
│       ├── train.py
│       └── evaluate.py
├── notebooks/
│   └── imagenette_distillation.ipynb
└── tests/
```

The final interface should support a small number of commands to configure, train, resume, and evaluate an experiment. Device selection should use CUDA when available and support a CPU fallback; the repository should not require paid cloud infrastructure. This structure is a plan, not the current repository layout.

### E. Keep paper implementations independently manageable

The repository is intended to give each paper its own development branch. Completed paper implementations can be reviewed and merged into `main` when their code, methodology, and results are documented.

## 8. Checkpointing and reproducibility features implemented

The notebook includes:

- Python, NumPy, CPU Torch, and CUDA Torch RNG-state capture and restoration.
- Separate DataLoader generator states, including reset for a genuinely new training stage.
- Atomic checkpoint writes using a temporary file and `os.replace`.
- Separate stage directories for the teacher and each student.
- `latest.pt` for epoch-boundary resumption and `best_model.pt` for evaluation.
- Model, optimizer, scheduler, gradient-scaler, epoch, best-metric, history, configuration, RNG, and DataLoader generator state in the resumable checkpoint.
- Stage compatibility checks before loading a checkpoint.

Resumption is designed around completed epoch boundaries. It does not restore an interrupted minibatch within an epoch. Exact cross-hardware reproducibility is not guaranteed, even when RNG states are restored.

## 9. Running the current notebook

The current implementation is a Google Colab-oriented notebook and was trained using a Colab Tesla T4 GPU. It mounts Google Drive for persistent experiment artifacts and downloads the Imagenette archive if it is not already present.

The clean committed notebook has its outputs and execution counts cleared. Its final cell invokes the teacher training stage. The remaining student stages are launched by explicit calls to `run_training_stage`; the KD calls must receive the trained, frozen `teacher_model` argument. For an interrupted stage, use the checkpoint-resume path (`resume=True`) rather than starting a new stage over existing checkpoint files.

The notebook writes data and experiment configuration JSON files and checkpoints to the configured work directory in Google Drive. Dataset archives, extracted images, checkpoints, and run outputs are intentionally not tracked in Git. The notebook is not yet a one-command, environment-pinned package; dependency specification and the modular CLI are part of the planned refactor.

## 10. Research record

The initial run is preserved because it answers a useful engineering and experimental question: what happens when a from-scratch teacher is trained on this dataset and used to supervise smaller students? In this particular run, the answer was that the teacher did not outperform the supervised student, and neither KD objective improved the student's best internal-validation accuracy.

That is not a universal conclusion about distillation. It is a reason to revisit teacher quality, regularisation, experimental controls, and evaluation before making a stronger claim.
