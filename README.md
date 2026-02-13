# Enhanced MRI Brain Tumor Classification using Multi-Feature Engineering and Explainable AI (SHAP)

This project implements a complete **multi-class brain tumor classification system** capable of identifying four tumor categories from MRI images:

- **Glioma**
- **Meningioma**
- **Pituitary Tumor**
- **No Tumor**

The system leverages **advanced feature engineering**, multiple machine-learning classifiers, and **explainable AI (SHAP)** to provide transparent, interpretable medical predictions.

---

## 📌 Overview
This work focuses on building a reliable brain tumor detection pipeline using classical machine‑learning methods combined with handcrafted feature extraction techniques. The goal is to produce a lightweight, interpretable alternative to deep learning models.

---

## 🧩 Feature Engineering Techniques

The system extracts rich discriminative features using:

### 🔹 **Discrete Wavelet Transform (DWT)**
Captures multi‑resolution spatial and frequency information.

### 🔹 **Fast Fourier Transform (FFT)**
Extracts frequency‑domain signatures of tumor regions.

### 🔹 **Gabor Filters**
Used for texture‑based analysis of MRI structures.

### 🔹 **Local Binary Patterns (LBP)**
Encodes local texture variations and edge patterns.

### 🔹 **Gray Level Run Length Matrix (GLRLM)**
Generates statistical texture descriptors from gray‑level distributions.

---

## ⚙️ Processing & Data Handling
- Modular feature loading functions.
- Automated batch processing for large datasets.
- Clean, reusable code structure for experimentation.
- Flexible integration of different feature pipelines.

---

## 🤖 Machine Learning Models

Multiple classifiers were trained and benchmarked:

- **Support Vector Machine (SVM)** — Linear & RBF  
- **K‑Nearest Neighbors (KNN)**
- **Random Forest**
- **Logistic Regression**

**Stratified Train/Test splitting** ensures balanced class representation.

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **Specificity**
- **F1‑Score**
- **Confusion Matrix Analysis**

A full comparative study is included to analyze the effectiveness of each feature extraction method.

---

## 🔍 Explainable AI — SHAP

To enhance interpretability:

- Integrated **SHAP (SHapley Additive Explanations)**  
- Generated **global feature importance** graphs  
- Visualized model predictions to explain *why* the classifier made a given decision  
- Supports transparent medical-AI deployment

---

## 📁 Automated Logging

All experimental results (metrics, parameters, comparisons) are logged automatically into:

- **Excel (.xlsx)**
- **CSV (.csv)**

This ensures reproducibility and structured reporting.

---

## 📂 Project Structure

/src dwt.py fft.py gabor.py glrlm.py lbp.py svc.py
/data (Place MRI images here)
README.md

---

## 🚀 Future Enhancements
- Dataset augmentation & normalization pipelines  
- Hybrid models combining handcrafted + deep features  
- Cross‑validation & hyperparameter tuning  
- SHAP-based feature selection  

---

## 📜 License
This project is open for educational and research purposes.

---

## 👤 Author
Developed by **Mariam Mohamed**
