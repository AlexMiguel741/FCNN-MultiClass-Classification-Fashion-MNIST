# Fashion-MNIST Classification with a Fully Connected Network

A minimal neural network baseline for classifying **10 clothing categories** (T-shirt, trouser,
sneaker, bag...) from 28×28 grayscale images, built step by step in TensorFlow/Keras.

## Results

- **~85.7% validation accuracy** with a very small model: **7,960 parameters**
- Most errors are between visually similar classes (shirt / pullover / coat)

## What it covers

- Data loading and normalisation (60,000 training images)
- Model built with the Keras functional API: `Flatten` → `Dense(10, sigmoid)` → `Dense(10, softmax)`
- Training with Adam and categorical cross-entropy, train/validation monitoring
- An equivalent `Sequential` formulation for comparison

The goal is a clean, well-understood baseline. Natural next step: a convolutional network,
like the ones in my [aerial image](https://github.com/AlexMiguel741/CNN-MultiClass-Classification-UC-Merced-Land-Use-Transfer-Learning)
and [CAM localization](https://github.com/AlexMiguel741/Mask-supervised-object-localization-GAP-CAM) projects.

## Stack

Python · TensorFlow / Keras · NumPy · Matplotlib

## Files

- [`FashionMNIST.ipynb`](FashionMNIST.ipynb): data, model, training, results
