import customtkinter as ctk
import tkinter as tk
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


# --- Model & encoding setup ---
pool7 = torch.nn.AvgPool2d(4)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
STATE_PATH = "snn_mnist_final_poission.pth"
TAU, HIDDEN, T_STEPS = 12.53, 203, 186
BETA = torch.tensor(np.exp(-1.0 / TAU), dtype=torch.float32, device=DEVICE)

from snntorch import surrogate, spikegen
import snntorch as snn

class SNN(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta1, beta2):
        super().__init__()
        self.fc1  = torch.nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=surrogate.fast_sigmoid())
        self.fc2  = torch.nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=surrogate.fast_sigmoid())
        self.bias1 = torch.nn.Parameter(torch.tensor(0.2, device=DEVICE))

    def forward(self, x, T):
        B = x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec = torch.zeros(T, B, self.fc2.out_features, device=DEVICE)
        for t in range(T):
            cur1 = self.fc1(x[t]) + self.bias1
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec[t] = spk2
        return spk2_rec

# Load trained model
model = SNN(49, HIDDEN, 10, BETA, BETA).to(DEVICE)
model.load_state_dict(torch.load(STATE_PATH, map_location=DEVICE))
model.eval()

def alpha_kernel(L=50, tau_r=1.0, tau_f=5.0, dt=1.0):
    t = torch.arange(0, L*dt, dt, device=DEVICE)
    return (t/tau_r)*torch.exp(1 - t/tau_r)*torch.exp(-t/tau_f)

# Encode 28x28 image to spikes
def encode_canvas_image(image, T=T_STEPS):
    img = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,28,28]
    pooled = pool7(img).view(1, -1)
    pooled = pooled / (pooled.max() + 1e-8)
    return spikegen.rate(pooled, num_steps=T)


# --- GUI setup ---
class DigitApp:
    def __init__(self, root):
        self.root = root
        self.canvas_size = 280  # pixels
        self.grid_size = 28
        self.cell_size = self.canvas_size // self.grid_size
        self.drawing = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Split into left/right panels
        self.main_frame = ctk.CTkFrame(root)
        self.main_frame.pack(padx=10, pady=10, fill="both")

        self.left_panel = ctk.CTkFrame(self.main_frame)
        self.left_panel.pack(side="left", fill="y")


        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.pack(side="right", fill="both", expand=True)
        self.title_label = ctk.CTkLabel(
            self.right_panel,
            text=f"α‑Kernel Filtered Waveforms",
            font=("Helvetica", 16, "bold"),
            anchor="center"
        )
        self.title_label.pack(pady=(10, 5))  # Adjust padding as needed


        # Drawing canvas on left
        canvas_frame = ctk.CTkFrame(self.left_panel)
        canvas_frame.pack(padx=5, pady=5)
        self.canvas = tk.Canvas(canvas_frame, width=self.canvas_size, height=self.canvas_size, bg='black',
                                highlightthickness=0)
        self.canvas.pack()

        # Draw grid lines
        for i in range(self.grid_size):
            self.canvas.create_line(i*self.cell_size, 0, i*self.cell_size, self.canvas_size, fill='gray')
            self.canvas.create_line(0, i*self.cell_size, self.canvas_size, i*self.cell_size, fill='gray')

        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        btn_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        btn_frame.pack(pady=10)

        predict_btn = ctk.CTkButton(btn_frame, text="Predict & Plot", command=self.run_prediction)
        predict_btn.pack(side="left", padx=10)

        clear_btn = ctk.CTkButton(btn_frame, text="Clear", command=self.clear_canvas)
        clear_btn.pack(side="right", padx=10)

        self.prediction_label = ctk.CTkLabel(self.left_panel, text="Prediction: —", font=("Helvetica", 16, "bold"))
        self.prediction_label.pack(pady=(5, 0))

        self.figure = Figure(figsize=(5, 8), constrained_layout=False)
        self.axes = [self.figure.add_subplot(10, 1, i + 1) for i in range(10)]
        for ax in self.axes:
            ax.set_ylim(0, 1.2)
            ax.grid(True)
        self.axes[4].set_xlabel("Time step")
        self.figure.suptitle("α‑Kernel Filtered Waveforms", fontsize=16, y=1.02)
        # self.figure.tight_layout(pad=2.5)
        self.figure.subplots_adjust(hspace=1)  # Increase spacing here

        self.canvas_fig = FigureCanvasTkAgg(self.figure, master=self.right_panel)
        self.canvas_fig.draw()
        self.canvas_fig.get_tk_widget().pack(padx=5, pady=5, fill="both", expand=True)
        self.mode_toggle = ctk.CTkSwitch(
            self.left_panel,
            text="Dark Mode",
            command=self.toggle_mode
        )

        # ─── Probability Bar Chart ───────────────────────────────
        self.prob_fig = Figure(figsize=(5, 2.5), tight_layout=True)
        self.prob_ax = self.prob_fig.add_subplot(1, 1, 1)
        # initialize blank chart
        self.prob_ax.bar(range(10), [0] * 10)
        self.prob_ax.set_xticks(range(10))
        self.prob_ax.set_xlabel("Digit", fontsize=10)
        self.prob_ax.set_ylabel("Probability (%)", fontsize=10)
        self.prob_ax.set_ylim(0, 100)
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, master=self.left_panel)
        self.prob_canvas.draw()
        self.prob_canvas.get_tk_widget().pack(padx=5, pady=(0, 10), fill="x")

        self.mode_toggle.pack(side="bottom", pady=(10, 10))
        self.mode_toggle.select()  # Default to dark mode
    def paint(self, event):
        x, y = event.x // self.cell_size, event.y // self.cell_size
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                xi, yi = x + dx, y + dy
                if 0 <= xi < self.grid_size and 0 <= yi < self.grid_size:
                    self.drawing[yi, xi] = 1.0
                    x1, y1 = xi * self.cell_size, yi * self.cell_size
                    self.canvas.create_rectangle(x1, y1, x1 + self.cell_size, y1 + self.cell_size, fill='white',
                                                 outline='')

    def toggle_mode(self):
        if self.mode_toggle.get() == 1:
            ctk.set_appearance_mode("dark")
            self.mode_toggle.configure(text="Dark Mode")
        else:
            ctk.set_appearance_mode("light")
            self.mode_toggle.configure(text="Light Mode")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.drawing = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.title_label.configure(
            text=f"α‑Kernel Filtered Waveforms")
        for i in range(self.grid_size):
            self.canvas.create_line(i*self.cell_size, 0, i*self.cell_size, self.canvas_size, fill='gray')
            self.canvas.create_line(0, i*self.cell_size, self.canvas_size, i*self.cell_size, fill='gray')

    def run_prediction(self):
        encoded = encode_canvas_image(self.drawing).to(DEVICE)  # shape [T, 1, 49]
        spk2 = model(encoded, T_STEPS)
        out_sum = spk2.sum(dim=0)  # [1,10]
        pred = out_sum.argmax(dim=1).item()
        print(f"Prediction: {pred}")

        # Plot waveforms
        kernel = alpha_kernel().view(1,1,-1)
        ap_waveforms = []
        for c in range(10):
            spikes = spk2[:, 0, c].view(1, 1, -1)
            ap = F.conv1d(spikes, kernel.to(DEVICE), padding=kernel.size(-1)//2)
            ap_waveforms.append(ap.view(-1).detach().cpu().numpy())

        self.prediction_label.configure(text=f"Prediction: {pred} — Confidence: {np.random.randint(50, 101)}%")


        fig, axes = plt.subplots(10, 1, figsize=(6, 20), sharex=True, sharey=True)
        for c in range(10):
            axes[c].plot(ap_waveforms[c])
            axes[c].set_title(f"Class {c}")
            axes[c].set_ylim(0, 1.2)
            axes[c].set_ylabel("Filtered spike")
            axes[c].grid(True)
        axes[-1].set_xlabel("Time step")
        # fig.suptitle(f"α‑Kernel Filtered Waveforms — Your Digit (Pred: {pred})", fontsize=16, y=1.02)
        plt.show()

        # Clear existing plots
        # Clear and configure subplots
        for ax in self.axes:
            ax.clear()
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 50, 100, 150])  # optional: reduce x ticks
            ax.set_yticks([0, 1])  # optional: reduce y ticks
            ax.tick_params(labelsize=8)
            ax.grid(True)

        # Plot and format
        for c in range(10):
            self.axes[c].plot(ap_waveforms[c])
            self.axes[c].set_title(f"Class {c}", fontsize=9, pad=2)
            self.axes[c].set_ylabel("")  # Remove individual labels for cleanliness

        # Shared y-label in the center of figure
        # self.figure.text(0.04, 0.5, "Filtered spike", va='center', rotation='vertical', fontsize=12)

        # Shared x-label at bottom
        self.axes[-1].set_xlabel("Time step", fontsize=10)

        # Subplot spacing
        # self.figure.subplots_adjust(hspace=0.5, left=0.1, right=0.95, top=0.88, bottom=0.08)
        # Add a shared y-label in the center of the figure
        self.figure.text(0.01, 0.5, "Filtered spike", va='center', rotation='vertical', fontsize=12)

        # Adjust subplot spacing to prevent overlap
        #self.figure.subplots_adjust(left=0.2, right=0.95, top=0.88, bottom=0.08)
        # Add a title above the graph in customtkinter
        self.title_label.configure(
            text=f"α‑Kernel Filtered Waveforms — Your Digit (Pred: {pred})"
        )

        out_sum = spk2.sum(dim=0).squeeze()  # shape [10]
        probs = F.softmax(out_sum, dim=0)  # torch tensor [10] summing to 1

        # update textual label
        pred_idx = int(probs.argmax().item())
        pred_prob = float(probs[pred_idx] * 100)
        self.prediction_label.configure(
            text=f"Prediction: {pred_idx}  (Confidence: {pred_prob:.1f}%)"
        )

        # ─── UPDATE BAR CHART ───────────────────────────────────
        pct = (probs.detach().cpu().numpy() * 100)
        self.prob_ax.clear()
        bars = self.prob_ax.bar(range(10), pct)
        self.prob_ax.set_xticks(range(10))
        self.prob_ax.set_xlabel("Digit", fontsize=10)
        self.prob_ax.set_ylabel("Probability (%)", fontsize=10)
        self.prob_ax.set_ylim(0, 100)
        # annotate each bar
        for i, v in enumerate(pct):
            self.prob_ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8)
        self.prob_canvas.draw()


        self.canvas_fig.draw()


# ─── Run App ─────────────────────────────────────────────────────
if __name__ == "__main__":
    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    root.title("Draw a Digit (28x28)")
    app = DigitApp(root)
    root.mainloop()
