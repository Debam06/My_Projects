import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageGrab
import math

class ShapeDrawer(tk.Canvas):
    def __init__(self, master):
        super().__init__(master, bg="white", width=500, height=500)
        self.bind("<B1-Motion>", self.draw)
        self.bind("<ButtonRelease-1>", self.reset_prev)
        self.pack(pady=10)
        self.prev_x, self.prev_y = None, None
        
        # Label to display detected shape
        self.shape_label = tk.Label(master, text="", font=("Arial", 16), fg="blue")
        self.shape_label.pack(pady=5)
    
    def draw(self, event):
        if self.prev_x is not None and self.prev_y is not None:
            self.create_line(self.prev_x, self.prev_y, event.x, event.y, fill="black", width=3)
        self.prev_x, self.prev_y = event.x, event.y

    def reset_prev(self, event):
        self.prev_x, self.prev_y = None, None

    def clear_canvas(self):
        self.delete("all")
        self.shape_label.config(text="")
        
    def save_drawing(self):
        # Get the canvas dimensions on the screen and grab that region.
        x = self.winfo_rootx()
        y = self.winfo_rooty()
        w = self.winfo_width()
        h = self.winfo_height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        img.save("drawing.png")

    def identify_shape(self):
        self.save_drawing()
        shape = detect_shape("drawing.png")
        self.shape_label.config(text=f"Detected Shape: {shape}")

def detect_shape(image_path):
    # Load the image from file.
    img = cv2.imread(image_path)
    if img is None:
        return "Error reading image"
    
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply thresholding - since the drawing is in black on white, use THRESH_BINARY_INV.
    ret, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find external contours.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return "No shape detected"
    
    # Select the largest contour to avoid small spurious artifacts.
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 100:  # Ignore very small contours
        return "No significant shape detected"
    
    # Calculate perimeter and approximate the contour.
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.04 * peri, True)
    vertex_count = len(approx)

    # Determine shape based on the number of vertices.
    if vertex_count == 3:
        shape = "Triangle"
    elif vertex_count == 4:
        # Distinguish between square and rectangle.
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            shape = "Square"
        else:
            shape = "Rectangle"
    elif vertex_count == 5:
        shape = "Pentagon"
    else:
        # For more than 5 vertices, check circularity.
        area = cv2.contourArea(largest_contour)
        circularity = 4 * math.pi * area / (peri * peri)
        if circularity > 0.8:
            shape = "Circle"
        else:
            shape = f"Polygon ({vertex_count} sides)"
    return shape

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Shape Recognition Drawing App")
    
    # Create the drawing area.
    drawer = ShapeDrawer(root)
    
    # Button for shape identification.
    btn_identify = tk.Button(root, text="Identify Shape", command=drawer.identify_shape, font=("Arial", 12))
    btn_identify.pack(side=tk.LEFT, padx=20, pady=10)
    
    # Button to clear the canvas.
    btn_clear = tk.Button(root, text="Clear", command=drawer.clear_canvas, font=("Arial", 12))
    btn_clear.pack(side=tk.RIGHT, padx=20, pady=10)
    
    root.mainloop()