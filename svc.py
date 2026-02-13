import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# ==========================================
# 
# ==========================================
features_root_dir = r'D:\Project\Features'  # المجلد اللي فيه ملفات npy

# 
filters_list = ['lbp', 'dwt', 'gabor', 'fft']

# 
classes_list = [
    "resized_images glioma",
    "resized_images Meningioma",
    "resized_images No Tumor", 
    "resized_images Pituitary"
]

display_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]


results_csv_file = 'results_progressive.csv'

# ==========================================
#
# ==========================================

def load_extracted_features(filter_name):
    """تحميل ملفات npy"""
    X = []
    y = []
    print(f"\n--- Loading Data for: {filter_name.upper()} ---")
    
    for label_idx, class_name in enumerate(classes_list):
        path = os.path.join(features_root_dir, filter_name, class_name, '*.npy')
        files = glob.glob(path)
        
        if files:
            print(f"  Class '{display_labels[label_idx]}': found {len(files)} files")
            for f in files:
                try:
                    vec = np.load(f)
                    if vec.ndim > 1: vec = vec.flatten()
                    X.append(vec)
                    y.append(label_idx)
                except:
                    pass
    return np.array(X), np.array(y)

def calculate_metrics_paper(y_true, y_pred):
    """حساب المقاييس"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    sens = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(len(cm)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
    
    avg_spec = np.mean(specificities)
    return acc, sens, prec, f1, avg_spec, cm

# ==========================================
# ==========================================

if __name__ == "__main__":
    print("Starting Classification Phase (Test Set Evaluation)...")


    if os.path.exists(results_csv_file):
        os.remove(results_csv_file)

    for filter_name in filters_list:

        X, y = load_extracted_features(filter_name)
        if len(X) == 0:
            print(f"Skipping {filter_name} (No Data)")
            continue


        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )


        clf = SVC(kernel='linear', C=1.0, random_state=42)
        print(f"  Training SVM on {filter_name.upper()}...")
        start_time = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        exec_time = time.time() - start_time


        acc, sens, prec, f1, spec, cm = calculate_metrics_paper(y_test, y_pred)


        print(f"\n=== Results for {filter_name.upper()} ===")
        print(f"Accuracy: {acc:.2%}, Sensitivity: {sens:.2%}, Precision: {prec:.2%}, F1-Score: {f1:.2%}, Specificity: {spec:.2%}")
        print(f"Execution Time: {exec_time:.4f}s")
        print("Confusion Matrix:")
        print(cm)


        df_entry = pd.DataFrame([{
            'Filter': filter_name.upper(),
            'Accuracy': acc,
            'Sensitivity': sens,
            'Precision': prec,
            'F1-Score': f1,
            'Specificity': spec,
            'Execution Time (s)': exec_time,
            'Confusion Matrix': cm.tolist()
        }])
        if not os.path.exists(results_csv_file):
            df_entry.to_csv(results_csv_file, index=False)
        else:
            df_entry.to_csv(results_csv_file, mode='a', header=False, index=False)


        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_labels, yticklabels=display_labels)
        plt.title(f"{filter_name.upper()} - Test Set\nTime: {exec_time:.2f}s | Acc: {acc:.1%}")
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

