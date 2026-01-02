"""
Waste Classification Training GUI
PyTorch + Tkinter
For TrashNet Dataset (6 classes: cardboard, glass, metal, paper, plastic, trash)
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from datetime import datetime

class WasteClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Waste Segregation Classifier - Training GUI")
        self.root.geometry("1200x800")
        self.root.configure(bg="#2c3e50")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.class_names = []
        self.is_training = False
        self.history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

        self.setup_gui()

    def setup_gui(self):
        # Title
        title = tk.Label(self.root, text="🗑️ Automated Waste Segregation System\nModel Training Dashboard",
                         font=("Arial", 18, "bold"), bg="#2c3e50", fg="#ecf0f1")
        title.pack(pady=10)

        tk.Label(self.root, text=f"Device: {self.device}", font=("Arial", 10), bg="#2c3e50", fg="#f39c12").pack()

        # Notebook
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tabs
        self.data_tab = ttk.Frame(notebook)
        self.train_tab = ttk.Frame(notebook)
        self.plot_tab = ttk.Frame(notebook)

        notebook.add(self.data_tab, text="📂 Dataset")
        notebook.add(self.train_tab, text="⚙️ Training")
        notebook.add(self.plot_tab, text="📊 Results")

        self.setup_data_tab()
        self.setup_train_tab()
        self.setup_plot_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, bg="#34495e", fg="white")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.update_time()

    def update_time(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.root.title(f"Waste Segregation Classifier - Training GUI | {current_time}")
        self.root.after(1000, self.update_time)

    def setup_data_tab(self):
        frame = ttk.Frame(self.data_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Directory selection
        dir_frame = ttk.LabelFrame(frame, text="Dataset Folder")
        dir_frame.pack(fill=tk.X, pady=5)

        self.dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.dir_var, width=80).pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)
        ttk.Button(dir_frame, text="Browse", command=self.browse_folder).pack(side=tk.RIGHT, padx=5)

        # Load button
        ttk.Button(frame, text="📁 Load TrashNet Dataset", command=self.load_dataset).pack(pady=10)

        # Info text
        self.info_text = scrolledtext.ScrolledText(frame, height=20)
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def setup_train_tab(self):
        frame = ttk.Frame(self.train_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Parameters
        params_frame = ttk.LabelFrame(frame, text="Training Parameters")
        params_frame.pack(fill=tk.X, pady=5)

        params = [
            ("Epochs", "epochs", 20),
            ("Batch Size", "batch", 32),
            ("Learning Rate", "lr", 0.001),
        ]

        self.vars = {}
        for i, (label, key, default) in enumerate(params):
            tk.Label(params_frame, text=label + ":").grid(row=i, column=0, sticky=tk.W, padx=10, pady=5)
            var = tk.IntVar(value=default) if key != "lr" else tk.DoubleVar(value=default)
            self.vars[key] = var
            spin = ttk.Spinbox(params_frame, from_=1 if key == "epochs" else 4, to=1000 if key == "batch" else 100,
                               textvariable=var, width=15)
            spin.grid(row=i, column=1, padx=10, pady=5)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(pady=20)

        self.start_btn = ttk.Button(btn_frame, text="▶️ Start Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        self.stop_btn = ttk.Button(btn_frame, text="⏹️ Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)

        ttk.Button(btn_frame, text="💾 Save Model", command=self.save_model).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="📂 Load Model", command=self.load_model).pack(side=tk.LEFT, padx=10)

        # Progress
        self.progress = ttk.Progressbar(frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)

        self.log_text = scrolledtext.ScrolledText(frame, height=15)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)

    def setup_plot_tab(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def log(self, msg):
        self.log_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_text.see(tk.END)
        self.status_var.set(msg)

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select TrashNet Dataset Folder")
        if folder:
            self.dir_var.set(folder)

    def load_dataset(self):
        path = self.dir_var.get()
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid dataset folder")
            return

        try:
            # Transforms
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            full_dataset = datasets.ImageFolder(path, transform=train_transform)
            self.class_names = full_dataset.classes

            # Split
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])
            val_set.dataset.transform = val_transform

            batch_size = self.vars["batch"].get()
            self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
            self.val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

            info = f"""
Dataset Loaded Successfully!
Classes: {', '.join(self.class_names)}
Total Images: {len(full_dataset)}
Training: {train_size}
Validation: {val_size}
Batch Size: {batch_size}
Image Size: 224x224
Augmentation: Enabled (training only)

Ready to train MobileNetV2!
            """
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
            self.log("Dataset loaded and ready!")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def build_model(self):
        self.model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace final layer
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, len(self.class_names))
        self.model.to(self.device)

        self.log("MobileNetV2 model built with transfer learning")

    def start_training(self):
        if self.train_loader is None:
            messagebox.showwarning("Warning", "Please load dataset first!")
            return

        if self.model is None:
            self.build_model()

        self.is_training = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.progress['value'] = 0

        threading.Thread(target=self.train_loop, daemon=True).start()

    def train_loop(self):
        try:
            optimizer = optim.Adam(self.model.classifier[1].parameters(), lr=self.vars["lr"].get())
            criterion = nn.CrossEntropyLoss()
            epochs = self.vars["epochs"].get()
            best_val_acc = 0.0

            self.log("Starting training (classifier only)...")

            for epoch in range(epochs):
                if not self.is_training:
                    break

                self.model.train()
                running_loss = correct = total = 0.0

                for i, (inputs, labels) in enumerate(self.train_loader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, pred = outputs.max(1)
                    total += labels.size(0)
                    correct += pred.eq(labels).sum().item()

                    self.root.after(0, lambda p=(i+1)/len(self.train_loader)*100: self.progress.config(value=p))

                train_acc = correct / total
                train_loss = running_loss / len(self.train_loader)

                # Validation
                self.model.eval()
                val_loss = val_correct = val_total = 0.0
                with torch.no_grad():
                    for inputs, labels in self.val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, pred = outputs.max(1)
                        val_total += labels.size(0)
                        val_correct += pred.eq(labels).sum().item()

                val_acc = val_correct / val_total
                val_loss /= len(self.val_loader)

                # === FIXED: Save best model + Confirmation Log ===
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    save_path = "best_model.pth"
                    torch.save(self.model.state_dict(), save_path)
                    self.root.after(0, lambda acc=val_acc: self.log(
                        f"🎉 NEW BEST MODEL SAVED! Validation Accuracy: {acc:.2%} → {save_path}"
                    ))

                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)

                msg = f"Epoch {epoch+1}/{epochs} → Train: {train_acc:.1%} | Val: {val_acc:.1%} | Best: {best_val_acc:.1%}"
                self.root.after(0, lambda m=msg: self.log(m))
                self.root.after(0, self.update_plots)

            # Final save and completion message
            torch.save(self.model.state_dict(), "final_model.pth")
            self.root.after(0, lambda: self.log("Training completed! Final model saved as 'final_model.pth'"))
            self.root.after(0, lambda: self.log(f"Best validation accuracy achieved: {best_val_acc:.2%}"))

        except Exception as e:
            self.root.after(0, lambda: self.log(f"Training error: {str(e)}"))
        finally:
            self.root.after(0, self.training_finished)

    def update_plots(self):
        self.ax1.clear()
        self.ax2.clear()

        epochs = range(1, len(self.history['train_acc']) + 1)

        self.ax1.plot(epochs, self.history['train_acc'], 'b-', label='Train Acc')
        self.ax1.plot(epochs, self.history['val_acc'], 'r-', label='Val Acc')
        self.ax1.set_title('Accuracy')
        self.ax1.legend()
        self.ax1.grid(True)

        self.ax2.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        self.ax2.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        self.ax2.set_title('Loss')
        self.ax2.legend()
        self.ax2.grid(True)

        self.canvas.draw()

    def stop_training(self):
        self.is_training = False
        self.log("Stopping training...")

    def training_finished(self):
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.progress['value'] = 100

    def save_model(self):
        if self.model is None:
            messagebox.showwarning("Warning", "No model to save")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pth", filetypes=[("PyTorch", "*.pth")])
        if path:
            torch.save(self.model.state_dict(), path)
            self.log(f"Model saved: {path}")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch", "*.pth")])
        if path and self.model:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.log(f"Model loaded: {path}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = WasteClassifierGUI(root)
    app.run()