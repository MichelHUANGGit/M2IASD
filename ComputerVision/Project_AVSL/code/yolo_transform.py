from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# Load the YOLO model
model = YOLO("yolov8n.pt")

# Perform inference on the image
results = model(r"data\test_images\2a8b5d63eae844d5af63d05fc139dd23.jpg")

# Get the image with bounding boxes
result_image = results[0].plot()

# Display the image using OpenCV
cv2.imshow('Image with Bounding Boxes', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
