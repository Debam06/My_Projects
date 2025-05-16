import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the PyTorch CNN model architecture.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # Input: 28x28; Output: 26x26
        self.pool = nn.MaxPool2d(2, 2)  # 26x26 -> 13x13
        self.fc1 = nn.Linear(32 * 13 * 13, 100)
        self.fc2 = nn.Linear(100, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))   # (batch, 32, 26, 26)
        x = self.pool(x)            # (batch, 32, 13, 13)
        x = x.view(x.size(0), -1)     # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        
        # Set canvas size (similar to MNIST scale but larger for drawing)
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        
        # Create a blank PIL image to mirror what's drawn on the canvas
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Bind mouse events: when left button is held down, draw.
        self.canvas.bind("<B1-Motion>", self.paint)
        
        # Buttons for prediction and clearing the canvas.
        predict_btn = tk.Button(root, text="Predict", command=self.predict)
        predict_btn.grid(row=1, column=0, sticky="ew", padx=10, pady=(0,10))
        
        clear_btn = tk.Button(root, text="Clear", command=self.clear)
        clear_btn.grid(row=2, column=0, sticky="ew", padx=10, pady=(0,10))
        
        # Label to display prediction results.
        self.result_label = tk.Label(root, text="Draw a Number", font=("Helvetica", 16))
        self.result_label.grid(row=3, column=0, padx=10, pady=10)

        # Load the trained PyTorch model from saved weights (model.pth).
        self.model = Net()
        self.model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
        self.model.eval()  # Set the model to evaluation mode

    def paint(self, event):
        # Brush size setting; adjust as needed.
        brush_size = 16
        x1, y1 = (event.x - brush_size//2), (event.y - brush_size//2)
        x2, y2 = (event.x + brush_size//2), (event.y + brush_size//2)
        
        # Draw circles on both the Tkinter canvas and the PIL image.
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        self.draw.ellipse([x1, y1, x2, y2], fill=0)
    
    def clear(self):
        # Clear the canvas and reinitialize the image.
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Draw a Number")
    
    def predict(self):
        # Preprocess the drawn image.
        # 1. Resize to 28x28 (MNIST input size).
        # 2. Invert colors (ensure the digit is white on a black background if that was used during training).
        # 3. Convert to numpy array.
        image_resized = self.image.resize((28, 28), Image.LANCZOS)
        image_inverted = ImageOps.invert(image_resized)
        img_array = np.array(image_inverted)
        
        # Normalize pixel values to [0, 1] and reshape to match PyTorch model input shape: [batch, channel, height, width].
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0) / 255.0
        
        # Do the prediction in a no-grad context to avoid unnecessary computation.
        with torch.no_grad():
            outputs = self.model(img_tensor)
            predicted = torch.argmax(outputs, dim=1).item()
        
        # Display the prediction.
        self.result_label.config(text=f"Predicted Digit: {predicted}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()