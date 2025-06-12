## Emotion-AI

## 🚀 Problem Statement

Emotion AI is a startup building real-time facial expression recognition systems, but they face a major challenge: a **small labeled dataset** (expression-annotated images) and a **large unlabeled dataset** (facial images without emotion labels).

We need to classify each face into one of **eight expression categories**:
- "Anger", 
- "Disgust", 
- "Fear", 
- "Happiness", 
- "Sadness", 
- "Surprise",
- "Neutral", 
- "Contempt"

---

## 📁 Dataset Overview

| Dataset Type     | Description                            | Size             |
|------------------|----------------------------------------|------------------|
| Labeled Dataset  | Grayscale facial images with emotion labels | 920 images (64x64) |
| Unlabeled Dataset| Grayscale facial images without labels | 13233 images (64x64) |

Both datasets were resized to 64×64 pixels and flattened into vectors of length 4096.

---

## Task folder structure

```
facial-expression-recognition/
│
├── 📁 data/                         
│   ├── ckextended.csv                       # Labeled dataset (e.g., CK+ subset)
│   └── lfw-deepfunneled/                    # Unlabeled dataset (e.g., LFW)
│
├── 📁 notebooks/                      
│   ├── train_model.ipynb                # experimenting different approaches to get the best model
│   ├── SVD_PCA.ipynb                    # Singular value decomposition and Principal component analysis
│   ├── preprocess.ipynb                 # Loading and analysing raw data
│   └── model_report.txt                 # classification report and best model parameters 
│
├── 📁 src/                           
│   ├── 
│   ├── preprocess.py              # Resize, grayscale, flatten
│   ├── svd_pca.py                 # SVD visualization on unlabeled data and PCA fitting and transformation
│   └── train.py                   # SVM training and evaluation
│
├── 📁 tests/                          
│   ├── ModelTraining_test.ipynb        # experimenting different approaches to get the best model
│   ├── SVD_PCA_test.ipynb              # Singular value decomposition and Principal component analysis
│   ├── preprocess_test.ipynb           # Loading and analysing raw data
│   └── report.txt                      # classification report and best model parameters 
│
├── requirements.txt                  # Project dependencies
├── README.md                         # Project overview
```

---

## 🧠 Approach Summary

### 🔹 Step 1: Preprocessing
- Resize all images to 64×64
- Convert to grayscale
- Flatten into 1D vectors of size 4096

### 🔹 Step 2: Dimensionality Reduction using PCA
- **Fit PCA on the large unlabeled dataset** to learn a general "face space"
- Select optimal `n_components = 68` based on explained variance using elbow plot
- **Transform the labeled data** using this PCA model to obtain lower-dimensional, meaningful features

### 🔹 Step 3: SVM Classification
- Train an SVM classifier on the labeled PCA-transformed features
- Perform hyperparameter tuning (kernel, C, gamma)

---

## 📉 Results

| Model No. | Description                            | Macro F1 | Accuracy | Best Params            |
|-----------|----------------------------------------|----------|----------|-------------------------|
| 1         | Basic SVM                              | 0.56     | 0.81     | C=100, gamma='scale'   |
| 2         | SVM + SMOTE                            | **0.59** | **0.82** | C=60, gamma='scale'    |
| 3         | SVM on Subset v1                       | 0.40     | 0.72     | C=70, gamma='scale'    |
| 4         | SVM on Subset v2                       | 0.51     | 0.62     | C=60, gamma='scale'    |
| 5         | Subset v2 + SMOTE                      | 0.48     | 0.59     | C=20, gamma='scale'    |
| 6         | Two-Model Hybrid                       | 0.40     | 0.74     | C=100 / C=40           |

- Best macro F1 is achieved with Model 2 (SMOTE only).
- Worst performing model (on F1 macro) is Model 3 and Model 6.

The model report can be found at [Model Report](https://github.com/hydraharish123/Emotion-AI/blob/main/notebooks/model_report.txt)



## 📌 Key Insights

- **SVD on unlabeled data** revealed that a small number of components (e.g., 68) explain most of the variance in facial images.
- **Using unlabeled data for PCA** significantly improved generalization and performance over fitting PCA only on the labeled set.
- **SVM with RBF kernel** provided the best results





