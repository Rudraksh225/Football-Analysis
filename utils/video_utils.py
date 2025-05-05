import cv2
import os

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames, output_video_path):

    # Ensure output_video_frames is not empty
    if not output_video_frames:
        raise ValueError("No frames to save in the video.")
    
    # Ensure the output path is valid
    if not output_video_path or not isinstance(output_video_path, str):
        raise ValueError("Invalid output video path.")
    
    height, width, _ = output_video_frames[0].shape
    
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release() 