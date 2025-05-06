import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
from scipy.signal import welch
from scipy.fft import fft
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import joblib

# === 参数设置 ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, "..", "data"))
model_dir = os.path.join(script_dir, "models")
os.makedirs(model_dir, exist_ok=True)
window_size = 100
step_size = 25
sampling_rate = 250

# === 特征提取函数（实时/训练统一） ===
def extract_features(fp1, fp2, eog_diff, sampling_rate=250):
    def compute(signal):
        features = [
            np.mean(signal), np.max(signal), np.min(signal), np.std(signal),
            np.ptp(signal), kurtosis(signal), skew(signal),
            np.max(np.abs(np.diff(signal))),
            np.sum(np.abs(np.diff(signal))),
            ((signal - np.mean(signal))**2).mean(),
            np.sum(np.sign(signal) != 0),
            np.count_nonzero(np.diff(np.sign(signal)))
        ]
        fft_vals = np.abs(fft(signal))[:len(signal)//2]
        freqs = np.fft.fftfreq(len(signal), d=1/sampling_rate)[:len(signal)//2]
        features.append(np.sum(fft_vals))
        features.append(freqs[np.argmax(fft_vals)])
        features.extend(fft_vals[:3])
        return features

    feats = compute(fp1) + compute(fp2) + compute(eog_diff)
    feats.append(np.mean(np.abs(fp1 - fp2)))
    feats.append(np.std(fp1 - fp2))
    feats.append(np.corrcoef(fp1, fp2)[0, 1])
    return feats

# === 加载数据 ===
all_trials = []
for file in os.listdir(data_dir):
    if file.endswith(".csv") and any(k in file for k in ["single", "double", "rest"]):
        df = pd.read_csv(os.path.join(data_dir, file))
        all_trials.append(df)

if not all_trials:
    raise FileNotFoundError("❌ No trial CSVs found.")

df_all = pd.concat(all_trials, ignore_index=True)
fp1_all = df_all["Fp1"].values
fp2_all = df_all["Fp2"].values
eog_all = df_all["EOG_diff"].values
labels = df_all["Marker"].values

# === 样本构造 ===
X_stage1, y_stage1 = [], []
X_stage2, y_stage2 = [], []

for i in range(0, len(fp1_all) - window_size, step_size):
    w_fp1 = fp1_all[i:i+window_size]
    w_fp2 = fp2_all[i:i+window_size]
    w_eog = eog_all[i:i+window_size]
    label = labels[i + window_size // 2]

    if label in ["single_blink", "double_blink"]:
        if np.max(np.abs(np.diff(w_eog))) < 40:
            continue

    feats = extract_features(w_fp1, w_fp2, w_eog)

    if label == "rest":
        X_stage1.append(feats)
        y_stage1.append("rest")
    elif label in ["single_blink", "double_blink"]:
        X_stage1.append(feats)
        y_stage1.append("blink")
        X_stage2.append(feats)
        y_stage2.append(label)

# === Stage 1: blink vs rest ===
X1, y1 = np.array(X_stage1), np.array(y_stage1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
le1 = LabelEncoder()
y1_train_enc = le1.fit_transform(y1_train)
y1_test_enc = le1.transform(y1_test)

clf1 = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=120, random_state=42))
clf1.fit(X1_train, y1_train_enc)
joblib.dump(clf1, os.path.join(model_dir, "blink_detector.pkl"))
joblib.dump(le1, os.path.join(model_dir, "blink_detector_labels.pkl"))
print("✅ [Stage 1] blink detector saved.")

y1_pred = clf1.predict(X1_test)
print("\n=== Stage 1: blink vs rest ===")
print(classification_report(y1_test_enc, y1_pred, target_names=le1.classes_))
cm1 = confusion_matrix(y1_test_enc, y1_pred)
ConfusionMatrixDisplay(cm1, display_labels=le1.classes_).plot(cmap="Blues")
plt.title("Stage 1 - Blink Detector")
plt.tight_layout()
plt.show()

# === Stage 2: single vs double ===
X2, y2 = np.array(X_stage2), np.array(y_stage2)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
le2 = LabelEncoder()
y2_train_enc = le2.fit_transform(y2_train)
y2_test_enc = le2.transform(y2_test)

clf2 = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=42))
clf2.fit(X2_train, y2_train_enc)
joblib.dump(clf2, os.path.join(model_dir, "blink_type_classifier.pkl"))
joblib.dump(le2, os.path.join(model_dir, "blink_type_labels.pkl"))
print("✅ [Stage 2] blink type classifier saved.")

y2_pred = clf2.predict(X2_test)
print("\n=== Stage 2: blink type ===")
print(classification_report(y2_test_enc, y2_pred, target_names=le2.classes_))
cm2 = confusion_matrix(y2_test_enc, y2_pred)
ConfusionMatrixDisplay(cm2, display_labels=le2.classes_).plot(cmap="Blues")
plt.title("Stage 2 - Blink Type Classifier")
plt.tight_layout()
plt.show()
