# Knowledge Distillation in Deep Learning: Effects of Alpha on Student Model Accuracy

This project explores response-based knowledge distillation (KD) and investigates how the choice of the distillation weight parameter, alpha, affects the performance of a student neural network. Built using Keras and tested on the MNIST handwritten digits dataset, this study seeks to understand how best to balance a student model’s learning from both the ground-truth labels and a pre-trained teacher model’s output.

This was an independent research project conducted under the guidance of Prof. Yunan Yang (August 2023).

## Overview

Knowledge distillation is a model compression technique where a large, high-performing teacher model transfers its learned knowledge to a smaller, more efficient student model. This project focuses on:

- Implementing response-based knowledge distillation
- Varying the alpha parameter that balances the student loss and distillation loss
- Conducting controlled experiments using the MNIST dataset
- Evaluating the student model’s performance across different alpha values

## Key Concepts

- Teacher Model: A complex CNN trained to ~99% accuracy on MNIST.
- Student Model: A lightweight CNN trained using a combination of real labels and teacher predictions.
- Alpha: A scalar in [0,1] that interpolates between student loss (ground-truth) and distillation loss (soft teacher labels).
- Distillation Loss: Cross-entropy between softened student and teacher logits.
- Student Loss: Cross-entropy between student predictions and hard labels.

## Research Question

How does varying the parameter alpha in knowledge distillation affect the accuracy of a student model compared to training it from scratch or purely from teacher predictions?

## Hypothesis

There exists an optimal range of alpha values that yields higher accuracy than training with hard labels alone, by leveraging the additional structure provided by the teacher’s soft predictions.

## Dataset

- MNIST handwritten digits
- 60,000 training samples and 10,000 test samples
- Image size: 28x28 grayscale
- Data preprocessing includes label perturbation (for noise analysis)

## Implementation Details

- Framework: TensorFlow / Keras
- Architecture:
  - Teacher: 2-layer CNN with 256 and 512 filters
  - Student: Lightweight CNN with 16 and 32 filters
- Core Class: Distiller, a subclass of keras.Model, handles custom loss computation
- Distillation temperature: Tunable (default = 3.0)

## Results

- The student model trained with alpha = 1.0 (hard labels only) achieved ~93.1% accuracy after 100 epochs.
- Teacher model achieved ~98.8% accuracy after 1000 epochs.
- Models trained with intermediate alpha values (~0.5) performed slightly better or comparably.
- Results suggest that knowledge distillation can enhance generalization under the right parameter tuning.

See `figures/accuracy_vs_alpha.png` for plotted results.

## Future Work

1. Explore different teacher model qualities and architectures
2. Try other datasets (e.g., Fashion MNIST, CIFAR-10)
3. Evaluate with metrics beyond accuracy (e.g., precision, recall, F1)
4. Study KD’s denoising effects under increasing noise levels
5. Theoretical analysis using mean-squared loss on synthetic datasets
