import gradio as gr
import numpy as np
import cv2
import tempfile
import tensorflow as tf
from tensorflow.keras.models import load_model
from google.colab import drive
drive.mount('/content/drive')

# Load your trained model from Google Drive
MODEL_PATH = "/content/drive/MyDrive/Celeb-DF/demo_best_model_efficientnet_lstm.keras"
model = load_model(MODEL_PATH)

SEQUENCE_LENGTH = 3
FRAME_SIZE = (128, 128)

def preprocess_frame(frame):
    frame = cv2.resize(frame, FRAME_SIZE)
    return frame / 255.0

def make_sequences(frames, sequence_length):
    # Overlapping, stride 1 for per-frame prediction
    sequences = []
    for i in range(len(frames) - sequence_length + 1):
        seq = np.array(frames[i:i+sequence_length])
        sequences.append(seq)
    return np.array(sequences)

def predict_on_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(preprocess_frame(frame))
    cap.release()
    if len(frames) < SEQUENCE_LENGTH:
        while len(frames) < SEQUENCE_LENGTH:
            frames.append(frames[-1])
    sequences = make_sequences(frames, SEQUENCE_LENGTH)
    if len(sequences) == 0:
        sequences = np.expand_dims(np.array(frames[:SEQUENCE_LENGTH]), axis=0)
    preds = model.predict(sequences, verbose=0)
    frame_probs = []
    for i in range(len(frames)):
        seq_indices = [j for j in range(len(preds)) if i == j + SEQUENCE_LENGTH - 1]
        if seq_indices:
            frame_probs.append(float(preds[seq_indices[0]][0]))
        else:
            frame_probs.append(float(preds[0][0]))  # fallback
    return frame_probs, frames

def detect_deepfake(video):
    frame_probs, frames = predict_on_video(video)
    temp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = temp_out.name
    height, width = frames[0].shape[:2]
    fps = 25  # Default or estimate from input
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for idx, (frame, prob) in enumerate(zip(frames, frame_probs)):
        label = 'FAKE' if prob > 0.5 else 'REAL'
        color = (0, 0, 255) if label == 'FAKE' else (0, 255, 0)
        annotated = (frame * 255).astype(np.uint8)
        cv2.putText(
            annotated,
            f'{label} ({prob*100:.1f}%)', (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
        )
        out.write(annotated)
    out.release()
    avg_prob = np.mean(frame_probs) * 100
    final_label = 'FAKE' if avg_prob > 50 else 'REAL'
    result_text = f"Prediction: {final_label} ({avg_prob:.2f}%)"
    return output_path, result_text

demo = gr.Interface(
    fn=detect_deepfake,
    inputs=gr.Video(label="🎥 Upload Video"),
    outputs=[gr.Video(label="🎯 Output Video"), gr.Textbox(label="🧠 Prediction")],
    title="💻 Deepfake Video Detector (EfficientNet+LSTM)",
    description="""Upload a short video. (EfficientNet+LSTM, trained on Celeb-DF, 3-frame sequences)."""
)

if __name__ == "__main__":
    demo.launch()
