# Fashion-MNIST Classification with a Fully Connected Neural Network

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Fashion--MNIST-green)

## Overview

Baseline fully connected network (FCNN) for 10-class classification of 28×28 grayscale clothing
images (Fashion-MNIST, Zalando Research). Implemented with the Keras functional API and an
input pipeline built on `tf.data`.

## Key Results

| Metric | Value |
| :-- | :-- |
| Validation accuracy | **85.7%** |
| Parameters | 7,960 |
| Main confusions | Shirt / Pullover / Coat (visually similar classes) |

## Model Architecture

```
Input 28×28
 └─ Flatten                    784
 └─ Dense(10, sigmoid)
 └─ Dense(10, softmax)
```

An equivalent `tf.keras.Sequential` implementation is included for comparison.

## Training Setup

| | |
| :-- | :-- |
| Data split | 50,000 train · 10,000 validation · 10,000 test |
| Input pipeline | `tf.data.Dataset`: normalization to [0, 1], one-hot encoding, batching |
| Optimizer | Adam, lr = 1e-3 |
| Loss | Categorical cross-entropy |
| Batch size | 32 |
| Epochs | 10 |

## Tech Stack

Python · TensorFlow / Keras · NumPy · Matplotlib

## Repository Structure

```
├── FashionMNIST.ipynb   # Data pipeline, model, training, predictions on the test set
└── README.md
```

## Related Projects

Convolutional models: [Aerial scene classification](https://github.com/AlexMiguel741/CNN-MultiClass-Classification-UC-Merced-Land-Use-Transfer-Learning) ·
[Weakly-supervised localization (CAM)](https://github.com/AlexMiguel741/Mask-supervised-object-localization-GAP-CAM)
