# 👕 FCNN Multi-Class Classification: Fashion MNIST

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A Deep Learning project implementing a **Fully Convolutional Neural Network (FCNN)** to classify clothing items from the **Fashion MNIST** dataset. This model achieves high accuracy in distinguishing between 10 different categories of apparel, addressing the challenges of intra-class similarity in grayscale images.

## 📌 Project Overview
This project uses Fashion MNIST, which contains images of clothing. It has the exact same format as the digit dataset (28x28 pixels), but it is much harder for the computer to solve. This makes it a better test to see how well our model can recognize complex shapes like shirts, coats, and sneakers

**Goal:** Accurately categorize 28x28 grayscale images into 10 distinct classes (e.g., T-shirts, Sneakers, Bags).

## 📂 The Dataset
- **Source:** [Fashion MNIST (Zalando Research)](https://github.com/zalandoresearch/fashion-mnist)
- **Training Set:** 60,000 images
- **Test Set:** 10,000 images
- **Dimensions:** 28x28 pixels (Grayscale)
- **Classes:**
  | Label | Description | | Label | Description |
  |---|---|---|---|---|
  | 0 | T-shirt/top | | 5 | Sandal |
  | 1 | Trouser | | 6 | Shirt |
  | 2 | Pullover | | 7 | Sneaker |
  | 3 | Dress | | 8 | Bag |
  | 4 | Coat | | 9 | Ankle boot |

## 🧠 Model Architecture
The solution uses a **Convolutional Neural Network (CNN)** optimized for image feature extraction [web:4]:

1.  **Input Layer:** Reshaped to `(28, 28, 1)` and normalized (0-1).
2.  **Convolutional Blocks:**
    -   `Conv2D` layers (32/64 filters) with `ReLU` activation for feature detection.
    -   `MaxPooling2D` for downsampling and reducing computational cost.
    -   `Dropout` (0.25) to prevent overfitting.
3.  **Classification Head:**
    -   `Flatten` layer to vectorise the feature maps.
    -   `Dense` hidden layer (128 units) with `ReLU`.
    -   **Output Layer:** 10 neurons with `Softmax` activation for multi-class probability [web:4].

## 📊 Performance
-   **Test Accuracy:** ~91-93% [web:1][web:3]
-   **Loss Function:** Categorical Crossentropy
-   **Optimizer:** Adam

*Sample Confusion Matrix:*
> The model successfully distinguishes distinct items (e.g., Trousers vs. Bags) but occasionally confuses visually similar classes like **Shirts** vs. **Coats** vs. **Pullovers** [web:1].

## 🛠️ Tech Stack
-   **Language:** Python
-   **Deep Learning Framework:** TensorFlow / Keras [web:1]
-   **Data Manipulation:** NumPy, Pandas
-   **Visualization:** Matplotlib, Seaborn

## 🚀 Getting Started

### Prerequisites
Make sure you have Python installed and the above libraries
