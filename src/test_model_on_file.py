import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import kurtosis, skew
from scipy.fft import fft
import os

# === 参数 ===
csv_path = "data/trial_10_double_blink.csv"  # 更换为任意 .csv 文件路径
model_path = "../models/blink_rest_classifier.pkl"
label_path = "../models/blink_rest_labels.pkl"
window_size = 50
step_size = 25

# === 特征提取函数 ===
def extract_features(signal):
    features = [
        np.mean(signal),
        np.max(signal),
        np.min(signal),
        np.std(signal),
        np.ptp(signal),
        kurtosis(signal),
        skew(signal)
    ]
    fft_vals = np.abs(fft(signal))[:len(signal)//2]
    freqs = np.fft.fftfreq(len(signal), d=1/250)[:len(signal)//2]
    features.append(np.sum(fft_vals))
    features.append(freqs[np.argmax(fft_vals)])
    features.extend(fft_vals[:3])
    return features

# === 载入数据 & 模型 ===
df = pd.read_csv(csv_path)
eog = df["EOG_diff"].values
model = joblib.load(model_path)
label_encoder = joblib.load(label_path)

# === 滑窗预测 ===
predictions = []
probs = []

for i in range(0, len(eog) - window_size, step_size):
    window = eog[i:i+window_size]
    feats = np.array(extract_features(window)).reshape(1, -1)
    pred = model.predict(feats)[0]
    prob = model.predict_proba(feats)[0]
    label = label_encoder.inverse_transform([pred])[0]
    predictions.append(label)
    probs.append(prob)

# === 打印结果示例 ===
for idx, label in enumerate(predictions[:10]):
    print(f"窗 {idx+1:02d}: Predicted → {label} | Prob: {np.round(probs[idx], 2)}")

# === 分类统计可视化 ===
from collections import Counter
counter = Counter(predictions)
plt.bar(counter.keys(), counter.values(), color='skyblue')
plt.title(f"Prediction Distribution for {os.path.basename(csv_path)}")
plt.ylabel("Window Count")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.show()
