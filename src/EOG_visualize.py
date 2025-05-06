import tkinter as tk
import time
import threading
import os
import csv
import random
import numpy as np
from pylsl import StreamInlet, resolve_stream
import glob

class BlinkRestCollector:
    def __init__(self, root, trials_per_class=30, duration=3, sampling_rate=250):
        self.root = root
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.label = tk.Label(root, text="Click 'Start' to begin", font=("Arial", 24))
        self.label.pack(pady=40)
        self.start_button = tk.Button(root, text="Start", font=("Arial", 18), command=self.start_collection)
        self.start_button.pack(pady=20)

        self.trial_list = (
            ["single_blink"] * trials_per_class +
            ["double_blink"] * trials_per_class +
            ["rest"] * trials_per_class
        )
        random.shuffle(self.trial_list)

        self.trial_index = 0
        self.save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        os.makedirs(self.save_dir, exist_ok=True)

        # Connect to EEG LSL stream
        print("Waiting for EEG stream...")
        self.inlet = StreamInlet(resolve_stream('type', 'EEG')[0])
        print("✅ EEG stream connected")

    def start_collection(self):
        self.start_button.config(state='disabled')
        # When running repeatedly, Remove old trial files
        # for label in ["single_blink", "double_blink", "rest"]:
        #     for f in glob.glob(os.path.join(self.save_dir, f"trial_*_{label}.csv")):
        #         os.remove(f)
        # print("🗑️ Old trial CSV files removed.")

        threading.Thread(target=self.run_trials).start()

    def run_trials(self):
        for label in self.trial_list:
            self.trial_index += 1
            instruction = {
                "single_blink": "👉 Please blink your RIGHT eye only",
                "double_blink": "👉 Please blink BOTH eyes once",
                "rest": "😌 Please stay relaxed and still"
            }[label]
            self.label.config(text=f"{instruction}\nTrial {self.trial_index}")
            print(f"🎬 Trial {self.trial_index} — Action: {label}")
            trial_data = []

            start_time = time.time()
            while time.time() - start_time < self.duration:
                sample, timestamp = self.inlet.pull_sample()
                if sample is None or len(sample) < 2:
                    continue
                fp1, fp2 = sample[0], sample[1]
                eog_diff = fp1 - fp2
                trial_data.append([timestamp, fp1, fp2, eog_diff, label])
                time.sleep(1 / self.sampling_rate)

            self.save_trial(trial_data, label, self.trial_index)
            self.label.config(text="Resting...")
            time.sleep(1.5)

        self.label.config(text="✅ All Trials Complete")

    def save_trial(self, data, label, index):
        filename = f"trial_{index:02d}_{label}.csv"
        full_path = os.path.join(self.save_dir, filename)

        #Data append and save
        file_exists = os.path.isfile(full_path)
        with open(full_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Timestamp", "Fp1", "Fp2", "EOG_diff", "Marker"])
            writer.writerows(data)

        # When running repeatedly, the original dataset is overwritten
        # with open(full_path, "w", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Timestamp", "Fp1", "Fp2", "EOG_diff", "Marker"])
        #     writer.writerows(data)
        # print(f"✅ Saved: {filename}")

# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Blink + Rest EOG Collector")
    app = BlinkRestCollector(root, trials_per_class=30, duration=3)
    root.mainloop()
