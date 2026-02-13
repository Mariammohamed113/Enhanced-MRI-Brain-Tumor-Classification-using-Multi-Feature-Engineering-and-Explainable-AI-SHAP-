# Enhanced-MRI-Brain-Tumor-Classification-using-Multi-Feature-Engineering-and-Explainable-AI-SHAP-
Designed and implemented a full multi-class brain tumor classification system (Glioma, Meningioma, Pituitary, No Tumor) using MRI images.
Key Contributions:
• Developed advanced feature extraction pipelines including:
    • Discrete Wavelet Transform (DWT)
    • Fast Fourier Transform (FFT)
    • Gabor Filters
    • Local Binary Patterns (LBP)
    • Gray Level Run Length Matrix (GLRLM)
• Built modular, reusable feature loaders and batch-processing pipelines for scalable dataset handling.
• Trained and benchmarked multiple machine-learning classifiers:
    • SVM (Linear & RBF), Random Forest, KNN, Logistic Regression
• Applied Stratified Train/Test Splitting to maintain class distribution.
• Evaluated models using comprehensive metrics:
    • Accuracy, Precision, Recall (Sensitivity), F1-Score, Specificity, Confusion Matrix
• Conducted comparative performance analysis across all feature extraction techniques.
• Integrated SHAP (SHapley Additive Explanations) to provide global feature importance and interpretability for medical AI decisions.
• Visualized SHAP outputs to enhance transparency in clinical prediction workflows.
• Automated logging of experimental results into Excel/CSV for structured reporting and reproducibility.
