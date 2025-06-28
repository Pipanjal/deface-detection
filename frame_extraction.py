import cv2
import os
from google.colab import drive
import tensorflow as tf
import torch
from tqdm import tqdm

drive.mount('/content/drive')

base_dir = '/content/drive/My Drive/Celeb-DF'
celeb_real_dir = os.path.join(base_dir, 'Celeb-real')
celeb_synthesis_dir = os.path.join(base_dir, 'Celeb-synthesis')
youtube_real_dir = os.path.join(base_dir, 'YouTube-real')

celeb_real_output_dir = os.path.join(base_dir, 'celeb-real-output')
celeb_synthesis_output_dir = os.path.join(base_dir, 'celeb-synthesis-output')
youtube_real_output_dir = os.path.join(base_dir, 'yt-output')

os.makedirs(celeb_real_output_dir, exist_ok=True)
os.makedirs(celeb_synthesis_output_dir, exist_ok=True)
os.makedirs(youtube_real_output_dir, exist_ok=True)

def extract_frames_from_videos(video_dir, output_dir, label):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]

    for video_name in tqdm(video_files, desc=f"Processing {label} videos"):
        video_path = os.path.join(video_dir, video_name)

        video_capture = cv2.VideoCapture(video_path)
        count = 0
        extracted = 0
        frame_logged = False

        while video_capture.isOpened():
            success, frame = video_capture.read()
            if not success:
                break

            frame_filename = f"{os.path.splitext(video_name)[0]}_{label}_frame{count}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)

            if os.path.exists(frame_path):
                count += 1
                continue

            # Optional: Log frame shape once
            if not frame_logged:
                print(f"Sample frame size from {video_name}: {frame.shape}")
                frame_logged = True

            cv2.imwrite(frame_path, frame)
            count += 1
            extracted += 1

        video_capture.release()
        print(f"Extracted {extracted} new frames from {video_name} (skipped existing).")

extract_frames_from_videos(celeb_real_dir, celeb_real_output_dir, 'celeb_real')
extract_frames_from_videos(celeb_synthesis_dir, celeb_synthesis_output_dir, 'celeb_synthesis')
extract_frames_from_videos(youtube_real_dir, youtube_real_output_dir, 'youtube_real')
