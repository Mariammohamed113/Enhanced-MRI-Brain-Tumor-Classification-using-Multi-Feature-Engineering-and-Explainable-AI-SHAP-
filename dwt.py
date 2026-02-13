# -----------------------------
# 1️⃣ Mount Google Drive
# -----------------------------
from google.colab import drive
drive.mount('/content/drive')
data_path = '/content/drive/MyDrive/Preprocessed_Dataset'

# -----------------------------
# 2️⃣ Imports
# -----------------------------
import os, cv2, numpy as np, time, pandas as pd
import pywt
from skimage.filters import gabor
from skimage.feature import local_binary_pattern
from scipy.fftpack import fft2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# GLRLM
!pip install mahotas
import mahotas

IMAGE_SIZE = (128,128)
classes = sorted(os.listdir(data_path))  # أسماء الورم
class_map = {c:i for i,c in enumerate(classes)}

# -----------------------------
# 3️⃣ Feature extraction functions
# -----------------------------
def extract_dwt(img): cA,_ = pywt.dwt2(img,'haar'); return cA.flatten()


feature_functions = {
    "DWT": extract_dwt,
}

# -----------------------------
# 4️⃣ Batch loading features
# -----------------------------
def load_features_batch(feature_func, batch_size=5000):
    X_batches, y_batches = [], []
    for c in classes:
        folder = os.path.join(data_path,c)
        img_files = os.listdir(folder)
        for i in range(0,len(img_files),batch_size):
            batch_files = img_files[i:i+batch_size]
            batch_features, batch_labels = [], []
            for f in batch_files:
                img_path = os.path.join(folder,f)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, IMAGE_SIZE)
                    batch_features.append(feature_func(img))
                    batch_labels.append(class_map[c])
            if batch_features:
                X_batches.append(np.array(batch_features))
                y_batches.append(np.array(batch_labels))
    X = np.vstack(X_batches)
    y = np.concatenate(y_batches)
    return X, y

# -----------------------------
# 5️⃣ Classifiers
# -----------------------------
def create_classifiers():
    classifiers = {
        "SVM": SVC(kernel='linear', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=500)
    }
    return classifiers

# -----------------------------
# 6️⃣ Evaluation function
# -----------------------------
def evaluate_model(clf, X_train, X_test, y_train, y_test, feat_name, clf_name, eval_type="Train/Test"):
    start_time = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    end_time = time.time()
    exec_time = end_time - start_time
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0) # Sensitivity
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Confusion matrix مع جميع الكلاسات حتى لو صفر
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
    tn = np.sum(cm) - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
    fp = cm.sum(axis=0) - np.diag(cm)
    specificity = np.mean(tn / (tn+fp))
    
    # Plot
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"{feat_name} - {clf_name} - {eval_type}")
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.show()
    
    return {
        "Feature": feat_name,
        "Classifier": clf_name,
        "Evaluation": eval_type,
        "Accuracy": acc,
        "Precision": precision,
        "Sensitivity": recall,
        "F1 Score": f1,
        "Specificity": specificity,
        "Execution Time": exec_time,
        "Model": clf
    }

# -----------------------------
# 7️⃣ Store results
# -----------------------------
results = []

# -----------------------------
# 8️⃣ Train/Test split (Stratified لضمان كل الكلاسات)
# -----------------------------
for feat_name, feat_func in feature_functions.items():
    print(f"\n=== Feature: {feat_name} | Train/Test Split ===")
    X, y = load_features_batch(feat_func)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)  # Stratified
    classifiers = create_classifiers()
    
    for clf_name, clf in classifiers.items():
        metrics = evaluate_model(clf, X_train, X_test, y_train, y_test, feat_name, clf_name, eval_type="Train/Test")
        results.append(metrics)
        print(f"{clf_name} Metrics: {metrics}")

# -----------------------------
# 9️⃣ Best Feature per Classifier for SHAP
# -----------------------------
best_features = {}
for clf_name in set([r["Classifier"] for r in results]):
    best = max([r for r in results if r["Classifier"]==clf_name], key=lambda x:x["Accuracy"])
    best_features[clf_name] = best

# -----------------------------
# 10️⃣ SHAP plots
# -----------------------------
for clf_name, best in best_features.items():
    feat_name = best["Feature"]
    clf_model = best["Model"]
    X, y = load_features_batch(feature_functions[feat_name])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    if clf_name in ["SVM","Random Forest","Logistic Regression"]:
        clf_model.fit(X_train, y_train)
        explainer = shap.Explainer(clf_model, X_train)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=[f"{feat_name}_{i}" for i in range(X_test.shape[1])])
        print(f"SHAP plot generated for {clf_name} on feature {feat_name}")

# -----------------------------
# 11️⃣ Save results to Excel
# -----------------------------
df_results = pd.DataFrame([{k:v for k,v in r.items() if k!="Model"} for r in results])
output_file = "/content/drive/MyDrive/Feature_Extraction_Results_Stratifed.xlsx"
df_results.to_excel(output_file, index=False)
print(f"\nAll results saved to: {output_file}")  asemdah
