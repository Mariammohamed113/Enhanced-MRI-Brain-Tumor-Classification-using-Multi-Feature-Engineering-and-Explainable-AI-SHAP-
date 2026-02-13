!pip install mahotas

import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import pywt
from tqdm import tqdm
import time
import mahotas

DATASET_ROOT = "/content/drive/MyDrive/Preprocessed_Dataset"
FEATURE_ROOT = "/content/drive/MyDrive/Features"
CLASSES = [
    "resized_images Meningioma",
    "resized_images No Tumor",
    "resized_images Pituitary",
    "resized_images glioma"
]

# ================== Feature Extraction Functions ==================
def extract_lbp_features(image, P=8, R=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P+3), density=True)
    return hist




    for theta in directions:
        # Determine offsets
        if theta == 0:
            dr, dc = 0, 1
        elif theta == np.pi/4:
            dr, dc = -1, 1
        elif theta == np.pi/2:
            dr, dc = -1, 0
        elif theta == 3*np.pi/4:
            dr, dc = -1, -1
        else:
            continue

        rows, cols = gray.shape
        glrlm = np.zeros((gray_levels, max(rows, cols)), dtype=np.float32)

        for i_row in range(rows):
            for j_col in range(cols):
                r, c = i_row, j_col
                val = gray[r, c]
                length = 0
                while 0 <= r < rows and 0 <= c < cols and gray[r, c] == val:
                    length += 1
                    r += dr
                    c += dc
                if length > 0:
                    glrlm[val, length-1] += 1

        glrlm_all.append(glrlm.flatten())

    feat_vec = np.hstack(glrlm_all)
    # Normalize
    feat_vec = feat_vec / (np.linalg.norm(feat_vec)+1e-6)
    return feat_vec

FEATURES = {
    "lbp": extract_lbp_features,

   
}

# ================== استخراج Features ==================
for feature_name, func in FEATURES.items():
    feature_folder = os.path.join(FEATURE_ROOT, feature_name)
    os.makedirs(feature_folder, exist_ok=True)
    print(f"\n================= Extracting {feature_name.upper()} ================")

    for cls in CLASSES:
        class_folder = os.path.join(DATASET_ROOT, cls)
        feature_class_folder = os.path.join(feature_folder, cls)
        os.makedirs(feature_class_folder, exist_ok=True)

        files = [f for f in os.listdir(class_folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        total_files = len(files)
        start_time = time.time()

        for i, f in enumerate(tqdm(files, desc=f"{feature_name.upper()} - {cls}")):
            img_path = os.path.join(class_folder, f)
            img = cv2.imread(img_path)
            if img is None:
                continue
            feat = func(img)
            np.save(os.path.join(feature_class_folder, f"{os.path.splitext(f)[0]}.npy"), feat)

            # تقدير الوقت المتبقي كل 100 صورة
            if (i+1) % 100 == 0 or i+1 == total_files:
                elapsed = time.time() - start_time
                remaining = (elapsed / (i+1)) * (total_files - (i+1))
                print(f"Processed {i+1}/{total_files} images, Estimated remaining time: {remaining/60:.2f} min")

        print(f"\u2705 Done {feature_name.upper()} for class '{cls}' in {time.time()-start_time:.2f} seconds")  dd