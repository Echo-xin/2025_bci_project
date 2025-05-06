import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from PIL import Image, ImageTk
from sklearn.metrics import accuracy_score

# 模拟无人机控制指令显示
class DroneController:
    def take_off(self):
        return "Drone is taking off!"

    def land(self):
        return "Drone is landing!"

# 模拟EOG信号分类
def classify_signal(signal):
    # 模拟EOG信号分类的过程，假设基于信号的幅度变化判断信号类型
    # 假设简单规则：信号波动幅度大于0.5为眨眼（blink），否则为静息（rest）
    amplitude = np.max(signal) - np.min(signal)
    if amplitude > 0.5:
        return "Single Blink" if random.random() > 0.5 else "Double Blink"
    else:
        return "Rest"

# 创建Tkinter GUI
class SignalGUI:
    def __init__(self, root, drone_controller):
        self.root = root
        self.drone_controller = drone_controller

        self.root.title("EOG Signal Recognition and Drone Control")

        # 创建显示信号波形图的框架
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.ax.set_title("Real-time EOG Signal")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Amplitude")
        self.line, = self.ax.plot([], [], lw=2)

        # 将matplotlib图形嵌入Tkinter界面
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(padx=10, pady=10)
        self.canvas.draw()

        # 创建分类结果标签
        self.classification_label = tk.Label(root, text="Signal Classification: ", font=("Arial", 14))
        self.classification_label.pack(pady=10)

        # 创建分类准确率标签
        self.accuracy_label = tk.Label(root, text="Real-time Accuracy: 0.00%", font=("Arial", 12))
        self.accuracy_label.pack(pady=10)

        # 创建按钮来控制无人机
        self.takeoff_button = tk.Button(root, text="Take Off", font=("Arial", 12), command=self.takeoff)
        self.takeoff_button.pack(pady=5)

        self.land_button = tk.Button(root, text="Land", font=("Arial", 12), command=self.land)
        self.land_button.pack(pady=5)

        # 创建显示无人机控制指令的标签
        self.drone_status_label = tk.Label(root, text="", font=("Arial", 14), fg="green")
        self.drone_status_label.pack(pady=20)

        # 创建显示无人机状态图标的标签
        self.drone_icon_label = tk.Label(root)
        self.drone_icon_label.pack(pady=10)

        # 模拟信号数据更新
        self.previous_classification = "Rest"
        self.update_signal()

    def update_signal(self):
        # 生成更接近实际EOG信号的波形（模拟眨眼信号的变化）
        time_data = np.arange(0, 100, 1)
        signal_data = np.sin(time_data / 10) + np.random.normal(0, 0.1, len(time_data))  # 信号噪声较小
        if random.random() > 0.5:  # 模拟眨眼信号
            signal_data[50:70] += np.sin(time_data[50:70] * 2) * 0.5  # 眨眼产生较大的幅度变化

        # 获取并显示分类结果
        classification = classify_signal(signal_data)
        self.classification_label.config(text=f"Signal Classification: {classification}")

        # 模拟分类准确率（假设真实结果随机，实际使用时需要通过对比分类器输出和真实标签来计算准确率）
        accuracy = random.uniform(90, 100)  # 假设真实分类器的准确率为90%到100%
        self.accuracy_label.config(text=f"Real-time Accuracy: {accuracy:.2f}%")

        # 更新信号图
        self.line.set_data(time_data, signal_data)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        # 每隔1秒更新一次信号和分类
        self.root.after(1000, self.update_signal)

    def takeoff(self):
        status = self.drone_controller.take_off()
        self.drone_status_label.config(text=status, fg="blue")
        self.update_drone_icon("takeoff.png")  # 显示起飞图标

    def land(self):
        status = self.drone_controller.land()
        self.drone_status_label.config(text=status, fg="red")
        self.update_drone_icon("land.png")  # 显示降落图标

    def update_drone_icon(self, icon_filename):
        # 更新无人机状态图标
        try:
            img = Image.open(icon_filename)
            img = img.resize((50, 50), Image.ANTIALIAS)
            img_tk = ImageTk.PhotoImage(img)
            self.drone_icon_label.config(image=img_tk)
            self.drone_icon_label.image = img_tk
        except FileNotFoundError:
            print(f"Icon file {icon_filename} not found!")

# 主程序
if __name__ == "__main__":
    drone_controller = DroneController()  # 创建模拟的无人机控制类
    root = tk.Tk()
    gui = SignalGUI(root, drone_controller)  # 初始化GUI界面
    root.mainloop()
