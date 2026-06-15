from ultralytics import YOLO
import torch
import cv2
import os

# Detect device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = YOLO('models/best_2.pt')

# Input video
video_path = r'test\test (2).mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video path
os.makedirs('output', exist_ok=True)
out_path = r'output\result_2.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

print("Processing video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on frame
    results = model.predict(frame, device=device, verbose=False)

    # Draw boxes on frame
    annotated_frame = results[0].plot()

    # Write frame to output video
    out.write(annotated_frame)

cap.release()
out.release()

print(f"Done! Output saved to: {out_path}")