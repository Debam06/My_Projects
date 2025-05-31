import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, Label, Button

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
labels = ["Unknown"] * 1000  # Placeholder
try:
    import requests
    labels = requests.get(LABELS_URL).text.split("\n")
except:
    print("Could not fetch ImageNet labels, using default.")

def classify_image(image_path):
    """Loads an image, processes it, and classifies it using MobileNetV2."""
    img = Image.open(image_path).convert("RGB")  # Convert to RGB
    img_transformed = transform(img).unsqueeze(0)  # Preprocess image for the model

    with torch.no_grad():
        outputs = model(img_transformed)
        _, predicted_class = outputs.max(1)  # Get the highest confidence prediction
        predicted_label = labels[predicted_class.item()]
    
    return f"Detected Object: {predicted_label}"

def upload_image():
    """Opens file dialog for user to select an image, displays it, and classifies it."""
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if file_path:
        try:
            image = Image.open(file_path)
            display_image = image.resize((400, 400))
            image_tk = ImageTk.PhotoImage(display_image)
            label_img.config(image=image_tk)
            label_img.image = image_tk  # Reference to prevent garbage collection

            result_text = classify_image(file_path)
            label_result.config(text=result_text)
        except Exception as e:
            label_result.config(text=f"Error processing image: {e}")

# Set up GUI window
root = tk.Tk()
root.title("Image Recognition App")
root.geometry("600x700")

btn = Button(root, text="Upload Image", command=upload_image, font=("Helvetica", 14))
btn.pack(pady=10)

label_img = Label(root)
label_img.pack(pady=10)

label_result = Label(root, text="", font=("Helvetica", 14))
label_result.pack(pady=10)

root.mainloop()