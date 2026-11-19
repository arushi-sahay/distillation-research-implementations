# Distillation Research Implementations

An open collection of independent implementations of influential research papers in knowledge distillation.

## About

Knowledge distillation covers several approaches for transferring useful information between machine-learning models. There is no single universal teacher-student architecture, loss function, or training procedure: each paper has its own assumptions and experimental context.

This repository studies those methods through independent implementations. The aim is to make the methods understandable through code, document the decisions made during implementation, and report experimental outcomes—including inconclusive or negative results—rather than presenting only favourable numbers.

## Current implementations

### Paper 01 — Distilling the Knowledge in a Neural Network (Hinton, Vinyals & Dean, 2015)

**Status:** Initial from-scratch training run completed. Evaluation and experimental redesign remain in progress.

The first Imagenette v2 experiment trained a ResNet-34 teacher and three ResNet-18 students. In this run, the hard-label student achieved the highest best internal-validation accuracy. The two distillation students did not improve on that baseline. The README for this paper documents the setup, results, limitations, and planned redesign.

- [Paper 01 methodology, results and next steps](papers/01-hinton-2015-distilling-knowledge/README.md)
- [Paper 01 notebook](papers/01-hinton-2015-distilling-knowledge/imagenette_distillation.ipynb)
- [Original paper](https://arxiv.org/abs/1503.02531)

## Guiding principles

### 1. Fidelity to original research

Each implementation should respect the original paper's problem formulation, methodology, and intended experimental context. Any adaptation should be identified rather than silently presented as an exact reproduction.

### 2. Independent implementations

There is no mandatory universal teacher-student architecture, distillation loss, or training framework. Each paper determines its own implementation requirements and architectural decisions.

### 3. Reproducibility and transparency

Implementations should document their methodology, experimental configuration, results, limitations, and deviations from the original work. Implementing a method, reproducing the paper's experiments, and extending its findings are distinct contributions and should be identified as such.

### 4. Report what the experiment shows

A result that does not support the initial hypothesis is still a result worth documenting. Claims should be limited to the tested setup and evidence; unfinished evaluations and known limitations should remain visible.

## Repository structure

```text
.
├── papers/
│   └── 01-hinton-2015-distilling-knowledge/
│       ├── README.md
│       └── imagenette_distillation.ipynb
└── README.md
```

Paper implementations may use different code structures. The first implementation currently uses a Colab notebook; a modular Python entry point is planned as part of its redesign.
