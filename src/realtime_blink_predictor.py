import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from pylsl import StreamInlet, resolve_stream
from collections import deque
import joblib
from scipy.stats import kurtosis, skew
from scipy.fft import fft

# === 特征提取函数（与训练完全一致） ===
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

class RealTimeBlinkVisualizer:
    def __init__(self, root, stage1_model_path, stage1_label_path, stage2_model_path, stage2_label_path, window_size=100):
        self.root = root
        self.root.title("实时眨眼两阶段分类器")
        self.window_size = window_size
        self.buffer_fp1 = deque(maxlen=window_size)
        self.buffer_fp2 = deque(maxlen=window_size)

        # 加载模型
        self.model_stage1 = joblib.load(stage1_model_path)
        self.le_stage1 = joblib.load(stage1_label_path)
        self.model_stage2 = joblib.load(stage2_model_path)
        self.le_stage2 = joblib.load(stage2_label_path)

        # LSL 连接
        print("⏳ 正在查找 EEG 流...")
        stream = resolve_stream('type', 'EEG')[0]
        self.inlet = StreamInlet(stream)
        print("✅ EEG 流连接成功！")

        # 主标签
        self.label = tk.Label(root, text="等待中...", font=("Arial", 36))
        self.label.pack(pady=10)

        # 图形界面
        fig, (self.ax_wave, self.ax_bar) = plt.subplots(2, 1, figsize=(5, 5), dpi=100)
        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=root)
        self.canvas.get_tk_widget().pack()

        self.line_wave, = self.ax_wave.plot([], [], lw=2)
        self.ax_wave.set_title("EOG_diff 波形")
        self.ax_wave.set_ylim(-300, 300)

        self.bar_labels = ['double_blink', 'rest', 'single_blink']
        self.bar_rects = self.ax_bar.bar(self.bar_labels, [0]*len(self.bar_labels), color="skyblue")
        self.ax_bar.set_ylim(0, 1.0)
        self.ax_bar.set_title("分类概率")

        self.update()

    def update(self):
        try:
            sample, _ = self.inlet.pull_sample(timeout=0.5)
            if sample is None or len(sample) < 2:
                self.root.after(40, self.update)
                return

            self.buffer_fp1.append(sample[0])
            self.buffer_fp2.append(sample[1])

            if len(self.buffer_fp1) == self.window_size:
                fp1 = np.array(self.buffer_fp1)
                fp2 = np.array(self.buffer_fp2)
                eog_diff = fp1 - fp2
                features = extract_features(fp1, fp2, eog_diff)

                if any(np.isnan(features)) or any(np.isinf(features)):
                    print("⚠️ 特征中存在 NaN 或 Inf，跳过该窗口")
                    self.root.after(40, self.update)
                    return

                prob_stage1 = self.model_stage1.predict_proba([features])[0]
                pred_stage1 = np.argmax(prob_stage1)
                label_stage1 = self.le_stage1.inverse_transform([pred_stage1])[0]

                if label_stage1 == 'rest':
                    final_label = 'rest'
                    probs = [0, 1, 0]  # double, rest, single
                else:
                    prob_stage2 = self.model_stage2.predict_proba([features])[0]
                    pred_stage2 = np.argmax(prob_stage2)
                    label_stage2 = self.le_stage2.inverse_transform([pred_stage2])[0]
                    final_label = label_stage2
                    probs = [0, 0, 0]
                    if label_stage2 == 'single_blink':
                        probs = [0, 0, 1]
                    else:
                        probs = [1, 0, 0]

                self.label.config(text=f"当前识别：{final_label.replace('_', ' ').title()}")
                self.line_wave.set_ydata(eog_diff)
                self.line_wave.set_xdata(np.arange(len(eog_diff)))
                self.ax_wave.set_xlim(0, len(eog_diff))

                for rect, p in zip(self.bar_rects, probs):
                    rect.set_height(p)

                self.canvas.draw()

        except Exception as e:
            print("⚠️ 实时预测错误：", e)

        self.root.after(40, self.update)


if __name__ == "__main__":
    root = tk.Tk()
    app = RealTimeBlinkVisualizer(
        root,
        stage1_model_path="D:/learning/oxford-Brookes/report/bci-m-test/.bci_test/src/models/blink_detector.pkl",
        stage1_label_path="D:/learning/oxford-Brookes/report/bci-m-test/.bci_test/src/models/blink_detector_labels.pkl",
        stage2_model_path="D:/learning/oxford-Brookes/report/bci-m-test/.bci_test/src/models/blink_type_classifier.pkl",
        stage2_label_path="D:/learning/oxford-Brookes/report/bci-m-test/.bci_test/src/models/blink_type_labels.pkl"
    )
    root.mainloop()
