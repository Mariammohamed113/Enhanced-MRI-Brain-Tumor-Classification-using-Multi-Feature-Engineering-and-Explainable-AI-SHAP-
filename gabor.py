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
from tqdm import tqdm

# ================== CONFIG ==================
FEATURE_ROOT = "/content/drive/MyDrive/Features"
FEATURE_NAME = "gabor"
feature_folder = os.path.join(FEATURE_ROOT, FEATURE_NAME)

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

# 🔹 تجربة سريعة: أول 100 ملف لكل Class
MAX_FILES_PER_CLASS = 100

# ================== UNIVERSAL LOADER ==================
def load_features(feature_folder, max_files=None):
    X, y = [], []

    for label, cls in enumerate(CLASSES):
        class_folder = os.path.join(feature_folder, cls)
        if not os.path.exists(class_folder):
            print(f"[WARNING] Folder not found: {class_folder}")
            continue

        files = [f for f in os.listdir(class_folder) if f.endswith(".npy")]
        if max_files:
            files = files[:max_files]

        print(f"[INFO] Loading {len(files)} files from {cls}...")
        for f in tqdm(files, desc=f"Loading {cls}", unit="file"):
            vec = np.load(os.path.join(class_folder, f))
            X.append(vec)
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    print(f"[INFO] Total features loaded: {X.shape}")
    return X, y

# ================== TRAINING ==================
X, y = load_features(feature_folder, max_files=MAX_FILES_PER_CLASS)

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
    plt.title(f"{FEATURE_NAME.upper()} - {model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
