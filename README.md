step-1
!pip install -q opencv-python-headless numpy matplotlib scikit-learn joblib

step-2
import os, zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn, joblib

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)

step-3
from google.colab import files
uploaded = files.upload()   # choose YOUR dataset.zip (Tomato, Potato, Onion folders)

with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
    zip_ref.extractall("dataset")

# Handle an extra wrapper folder
root = "dataset"
contents = os.listdir(root)
if len(contents) == 1 and os.path.isdir(os.path.join(root, contents[0])):
    root = os.path.join(root, contents[0])

print("Using folder:", root)
print("Classes found:", sorted(os.listdir(root)))   # should show ['Onion', 'Potato', 'Tomato']


step-4
DATASET_DIR = root
IMG_SIZE = 64
CLASSES = ["Onion", "Potato", "Tomato"]    # label 0, 1, 2

for class_name in CLASSES:
    folder = os.path.join(DATASET_DIR, class_name)
    if not os.path.isdir(folder):
        print("Missing folder:", folder)
        continue
    images = [f for f in os.listdir(folder)
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
    print(f"Class {class_name}: {len(images)} images")


step-5
sample_folder = os.path.join(DATASET_DIR, "Tomato")
sample_files = [f for f in os.listdir(sample_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
image = cv2.imread(os.path.join(sample_folder, sample_files[0]))

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Sample: Tomato")
plt.axis("off")
plt.show()

resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
features = resized.flatten().astype(np.float32) / 255.0
print("Original shape:", image.shape)
print("Resized shape:", resized.shape)
print("Number of features:", features.shape[0])   # 12288
print("Min/Max:", features.min(), features.max())

step-6
def preprocess(img_bgr):
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    small = cv2.resize(img, (16, 16)).astype(np.float32).flatten() / 255.0

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (~((s < 40) & (v > 200))).astype(np.uint8) * 255   # ignore white background
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [12, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-6)

    return np.concatenate([hist * 5, small]).astype(np.float32)

def load_dataset(dataset_dir):
    X, y = [], []
    for label, class_name in enumerate(CLASSES):
        folder = os.path.join(dataset_dir, class_name)
        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            img = cv2.imread(os.path.join(folder, filename))
            if img is None:
                print("Skipping unreadable image:", filename)
                continue
            X.append(preprocess(img))
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)

X, y = load_dataset(DATASET_DIR)
print("Feature matrix shape:", X.shape)
print("Label array shape:", y.shape)
for i, c in enumerate(CLASSES):
    print(f"  {i} = {c}: {(y == i).sum()} images")

step-7
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

step-8
from sklearn.svm import SVC

model = SVC(kernel="linear", probability=True, random_state=42)
model.fit(X_train, y_train)
print("Model training completed.")

step-9
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, probability=True, random_state=42))
model.fit(X_train, y_train)
print("Model training completed.")

step-10
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues", values_format="d")
plt.title("Tomato / Potato / Onion - Confusion Matrix")
plt.show()

step-11
wrong = np.where(y_pred != y_test)[0]
print(f"{len(wrong)} wrong out of {len(y_test)}")
if len(wrong):
    n = min(8, len(wrong))
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.8))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, wrong[:n]):
        ax.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE, 3)[..., ::-1])
        ax.set_title(f"True: {CLASSES[y_test[i]]}\nPred: {CLASSES[y_pred[i]]}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

step-12
up = files.upload()
fname = list(up.keys())[0]

test_img = cv2.imread(fname)
proba = model.predict_proba([preprocess(test_img)])[0]
pred = int(np.argmax(proba))

plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.title(f"Model says: {CLASSES[pred]} ({proba[pred]:.0%})")
plt.axis("off")
plt.show()

step-13
joblib.dump({"model": model, "classes": CLASSES, "img_size": IMG_SIZE,
             "accuracy": float(accuracy)}, "tomato_potato_onion_model.joblib")
files.download("tomato_potato_onion_model.joblib")
print("Model saved successfully.")

step-14
print("Total images :", len(y))
print("Train / Test :", len(X_train), "/", len(X_test))
print("Image size   :", IMG_SIZE, "x", IMG_SIZE)
print("Model        : Linear SVM")
print(f"Accuracy     : {accuracy:.2%}")
