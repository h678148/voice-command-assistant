
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import librosa
import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent

SR = 16000
N_SAMPLES = 16000
N_MELS = 64
HOP_LENGTH = 160
N_FFT = 400

CLASS_NAMES = ["up", "down", "left", "right"]

class SmallSpectrogramCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)

def pad_or_trim(x, n_samples=N_SAMPLES):
    x = x.astype(np.float32)
    if len(x) < n_samples:
        x = np.pad(x, (0, n_samples - len(x)), mode="constant")
    else:
        x = x[:n_samples]
    return x

def audio_to_mel_db(audio, sr):
    audio = audio.astype(np.float32)
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    audio = pad_or_trim(audio, N_SAMPLES)

    S = librosa.feature.melspectrogram(
        y=audio,
        sr=SR,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=SR/2,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    return S_db

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SmallSpectrogramCNN(n_classes=4).to(device)

# Expect checkpoint file in same folder as app.py
ckpt_path = PROJECT_ROOT / "cnn_4class_best.pt"
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

@torch.no_grad()
def predict(audio_input):
    if audio_input is None:
        return "No audio provided.", 0.0, 0.0, 0.0, {}

    sr, audio = audio_input
    audio = audio.astype(np.float32)

    t0 = time.perf_counter()

    t_pre0 = time.perf_counter()
    S_db = audio_to_mel_db(audio, sr)
    t_pre1 = time.perf_counter()

    t_inf0 = time.perf_counter()
    X = torch.from_numpy(S_db).unsqueeze(0).unsqueeze(0).float().to(device)
    logits = model(X)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy().squeeze()
    t_inf1 = time.perf_counter()

    pred_id = int(np.argmax(probs))
    pred_label = CLASS_NAMES[pred_id]
    confidence = float(probs[pred_id])

    preprocess_ms = (t_pre1 - t_pre0) * 1000.0
    inference_ms = (t_inf1 - t_inf0) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0

    prob_dict = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}

    return pred_label, confidence, preprocess_ms, inference_ms, total_ms, prob_dict

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(sources=["microphone", "upload"], type="numpy", label="Speak: up / down / left / right"),
    outputs=[
        gr.Label(label="Predicted command"),
        gr.Number(label="Confidence"),
        gr.Number(label="Preprocess latency (ms)"),
        gr.Number(label="Inference latency (ms)"),
        gr.Number(label="Total latency (ms)"),
        gr.JSON(label="Probabilities"),
    ],
    title="Voice Command Assistant",
    description="A small local speech command recognizer trained from scratch on Google Speech Commands.",
)

if __name__ == "__main__":
    demo.launch()
