import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import requests

# Load MobileNetV2 pretrained model
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Define image preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet normalization
])

# Load class labels from ImageNet
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = ["Unknown"] * 1000
try:
    labels = requests.get(LABELS_URL).text.split("\n")
except:
    print("Could not fetch ImageNet labels.")

def classify_image(image_path):
    """Loads an image, processes it, and classifies it using MobileNetV2."""
    img = Image.open(image_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_transformed)
        _, predicted_class = outputs.max(1)
        predicted_label = labels[predicted_class.item()]
    
    return f"Detected Object: {predicted_label}"

def upload_image():
    """Opens file dialog for user to select an image, displays it, and classifies it."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        try:
            image = Image.open(file_path)
            display_image = image.resize((400, 400))
            image_tk = ImageTk.PhotoImage(display_image)
            label_img.config(image=image_tk)
            label_img.image = image_tk

            label_status.config(text="Processing image...")
            root.update()

            result_text = classify_image(file_path)
            label_result.config(text=result_text)
            label_status.config(text="Analysis complete!")
        except Exception as e:
            label_result.config(text=f"Error: {e}")

# Set up GUI window with better styling
root = tk.Tk()
root.title("Image Recognition App")
root.geometry("700x800")
root.configure(bg="#2E2E2E")

# Header Frame
header = Frame(root, bg="#444", padx=10, pady=10)
header.pack(fill="x")

title = Label(header, text="📷 Image Recognition App", font=("Helvetica", 20, "bold"), bg="#444", fg="#fff")
title.pack()

# Upload Button
btn = Button(root, text="Upload Image", command=upload_image, font=("Helvetica", 14), bg="#008CBA", fg="white", padx=10, pady=5)
btn.pack(pady=20)

# Image Display Label
label_img = Label(root, bg="#2E2E2E")
label_img.pack(pady=10)

# Status Bar
label_status = Label(root, text="Awaiting image upload...", font=("Helvetica", 12), bg="#444", fg="#fff")
label_status.pack(pady=10, fill="x")

# Classification Result Label
label_result = Label(root, text="", font=("Helvetica", 16), bg="#2E2E2E", fg="#00FF00", justify="center", wraplength=600)
label_result.pack(pady=20)

root.mainloop()