import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# ================== CONFIG ==================
FEATURE_ROOT = "/content/drive/MyDrive/Features/glrlm"
CLASSES = [
    "resized_images Meningioma",
    "resized_images No Tumor",
    "resized_images Pituitary",
    "resized_images glioma"
]

MODELS = {
    "SVM": SVC(kernel='rbf', probability=True),
    "RandomForest": RandomForestClassifier(n_estimators=200),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "LogisticRegression": LogisticRegression(max_iter=500)
}

# ================== LOAD MERGED FILES ==================
X, y = [], []
SAMPLES_PER_CLASS = 50  # أول 50 sample لكل Class

for label, cls in enumerate(CLASSES):
    file_path = os.path.join(FEATURE_ROOT, f"{cls}.npy")
    if not os.path.exists(file_path):
        print(f"[WARNING] File not found: {file_path}")
        continue

    data = np.load(file_path)
    if SAMPLES_PER_CLASS:
        data = data[:SAMPLES_PER_CLASS]  # أول 50 sample
    X.extend(data)
    y.extend([label]*len(data))
    print(f"[INFO] Loaded {len(data)} samples from {cls}")

X = np.array(X)
y = np.array(y)
print(f"[INFO] Total features loaded: {X.shape}")

# ================== TRAINING ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

for model_name, model in MODELS.items():
    print(f"\n================ {model_name} =================")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"[RESULT] Accuracy: {acc*100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=CLASSES))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASSES,
                yticklabels=CLASSES)
    plt.title(f"GLRLM - {model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
