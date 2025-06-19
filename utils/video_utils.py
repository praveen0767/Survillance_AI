import cv2
import torch
import numpy as np
import os
from datetime import datetime

def preprocess_frame(frame, size=(128, 128)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, size)
    normalized = resized / 255.0
    return torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def save_anomaly_clip(frames, output_dir='clips/anomalies'):
    if not frames:
        raise ValueError("No frames to save.")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    clip_path = os.path.join(output_dir, f'anomaly_{timestamp}.avi')
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(clip_path, fourcc, 10.0, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()
    return clip_path

def generate_heatmap(input_frame, output_frame, size):
    diff = np.abs(input_frame - output_frame)
    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8) * 255
    diff = diff.astype(np.uint8)
    heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, size)
    return heatmap

def load_ucsd_dataset(data_path):
    frames = []
    for folder in sorted(os.listdir(data_path)):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            for file in sorted(os.listdir(folder_path)):
                if file.endswith('.tif'):
                    frame_path = os.path.join(folder_path, file)
                    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                    if frame is not None:
                        frame = cv2.resize(frame, (128, 128))
                        frame = frame / 255.0
                        frames.append(torch.tensor(frame, dtype=torch.float32).unsqueeze(0))
    return frames