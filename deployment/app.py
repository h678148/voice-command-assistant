import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import librosa
import gradio as gr
import matplotlib.pyplot as plt
import random


dark_css = """
body {
    background-color: #0f172a;
}

.gradio-container {
    background-color: #0f172a;
}

h1, h2, h3, h4, p, label {
    color: #e5e7eb !important;
}

button {
    background-color: #1f2937 !important;
    color: white !important;
    border: 1px solid #374151 !important;
}

textarea, input {
    background-color: #111827 !important;
    color: white !important;
    border: 1px solid #374151 !important;
}

.tab-nav button {
    background-color: #1f2937 !important;
    color: white !important;
}
"""


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "cnn_5class_E2_downsample_aug_1772993374_best.pt"

SR = 16000
N_SAMPLES = 16000
N_MELS = 64
HOP_LENGTH = 160
N_FFT = 400

CLASS_NAMES = ["up", "down", "left", "right", "unknown"]
SNAKE_COMMANDS = {"up", "down", "left", "right"}
CONF_THRESH = 0.55

class SmallSpectrogramCNN(nn.Module):
    def __init__(self, n_classes=5):
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SmallSpectrogramCNN(n_classes=5).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def pad_or_trim(x: np.ndarray, n_samples: int = N_SAMPLES) -> np.ndarray:
    x = x.astype(np.float32)
    if len(x) < n_samples:
        x = np.pad(x, (0, n_samples - len(x)), mode="constant")
    else:
        x = x[:n_samples]
    return x

def audio_to_mel_db(audio: np.ndarray, sr: int) -> np.ndarray:
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
        fmax=SR / 2,
        power=2.0,
    )
    S_db = librosa.power_to_db(S, ref=np.max).astype(np.float32)
    return S_db

@torch.no_grad()
def predict_command_breakdown(audio: np.ndarray, sr: int):
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

def gradio_predict(audio_input):
    if audio_input is None:
        return "No audio provided.", 0.0, 0.0, 0.0, 0.0, {}

    sr, audio = audio_input
    audio = audio.astype(np.float32)

    pred_label, conf, pre_ms, inf_ms, total_ms, prob_dict = predict_command_breakdown(audio, sr)
    return pred_label, conf, pre_ms, inf_ms, total_ms, prob_dict

GRID_H, GRID_W = 12, 12

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

def is_opposite(d1, d2):
    opposites = {("up", "down"), ("down", "up"), ("left", "right"), ("right", "left")}
    return (d1, d2) in opposites

def spawn_food(snake, rng):
    empty = [(r, c) for r in range(GRID_H) for c in range(GRID_W) if (r, c) not in snake]
    return rng.choice(empty) if empty else None

def new_game(seed=0):
    rng = random.Random(seed)
    start_r, start_c = GRID_H // 2, GRID_W // 2
    snake = [(start_r, start_c), (start_r, start_c - 1), (start_r, start_c - 2)]
    direction = "right"
    food = spawn_food(snake, rng)
    return {
        "snake": snake,
        "direction": direction,
        "food": food,
        "score": 0,
        "alive": True,
        "rng_seed": seed,
        "step": 0,
    }

def step_game(state, new_direction=None):
    if not state["alive"]:
        return state

    if new_direction is not None:
        if new_direction in DIRS and not is_opposite(state["direction"], new_direction):
            state["direction"] = new_direction

    dr, dc = DIRS[state["direction"]]
    head_r, head_c = state["snake"][0]
    new_head = (head_r + dr, head_c + dc)

    if not (0 <= new_head[0] < GRID_H and 0 <= new_head[1] < GRID_W):
        state["alive"] = False
        return state

    if new_head in state["snake"]:
        state["alive"] = False
        return state

    state["snake"] = [new_head] + state["snake"]

    if state["food"] is not None and new_head == state["food"]:
        state["score"] += 1
        rng = random.Random(state["rng_seed"] + state["step"] + 1)
        state["food"] = spawn_food(state["snake"], rng)
    else:
        state["snake"].pop()

    state["step"] += 1
    return state

def render_board(state):
    board = np.zeros((GRID_H, GRID_W), dtype=np.float32)

    if state["food"] is not None:
        fr, fc = state["food"]
        board[fr, fc] = 0.5

    for (r, c) in state["snake"][1:]:
        board[r, c] = 0.8

    hr, hc = state["snake"][0]
    board[hr, hc] = 1.0

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(board, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Score: {state['score']} | Alive: {state['alive']} | Dir: {state['direction']}")
    fig.tight_layout()

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb

def ui_reset():
    state = new_game(seed=0)
    return state, render_board(state), state["score"], state["alive"], state["direction"]

def ui_step(state):
    state = step_game(state)
    return state, render_board(state), state["score"], state["alive"], state["direction"]

def ui_turn(state, direction):
    state = step_game(state, new_direction=direction)
    return state, render_board(state), state["score"], state["alive"], state["direction"]

def voice_step_threshold(state, audio_input):
    if audio_input is None:
        return state, render_board(state), state["score"], state["alive"], state["direction"], "No audio"

    sr, audio = audio_input
    audio = audio.astype(np.float32)

    pred_label, conf, pre_ms, inf_ms, total_ms, prob_dict = predict_command_breakdown(audio, sr)

    applied = False

    if pred_label in SNAKE_COMMANDS and conf >= CONF_THRESH:
        state = step_game(state, new_direction=pred_label)
        applied = True
    else:
        state = step_game(state, new_direction=None)

    info = (
        f"Predicted: {pred_label} | Conf: {conf:.2f} | Applied: {applied} | "
        f"Latency (pre/inf/total): {pre_ms:.2f}/{inf_ms:.2f}/{total_ms:.2f} ms"
    )

    return state, render_board(state), state["score"], state["alive"], state["direction"], info


with gr.Blocks(css=dark_css) as demo:
    gr.Markdown("# 🎙️ Voice Command Assistant + 🐍 Snake Demo")
    gr.Markdown(
        "Tabs: 5-class classifier, manual Snake, and voice-controlled Snake. "
        f"Snake uses confidence threshold = **{CONF_THRESH}** and treats **unknown** as no-op."
    )

    with gr.Tabs():
        with gr.Tab("Voice classifier"):
            gr.Markdown("Speak one of: **up, down, left, right**. The model also supports **unknown**.")
            audio_in1 = gr.Audio(sources=["microphone", "upload"], type="numpy")

            out_label = gr.Label(label="Predicted command")
            out_conf = gr.Number(label="Confidence")
            out_pre = gr.Number(label="Preprocess latency (ms)")
            out_inf = gr.Number(label="Inference latency (ms)")
            out_total = gr.Number(label="Total latency (ms)")
            out_probs = gr.JSON(label="Probabilities")

            btn1 = gr.Button("Predict")
            btn1.click(
                fn=gradio_predict,
                inputs=[audio_in1],
                outputs=[out_label, out_conf, out_pre, out_inf, out_total, out_probs],
            )

        with gr.Tab("Snake (manual)"):
            game_state = gr.State(new_game(seed=0))

            board_img = gr.Image(label="Snake board", type="numpy")
            score_out = gr.Number(label="Score")
            alive_out = gr.Checkbox(label="Alive")
            dir_out = gr.Textbox(label="Direction")

            with gr.Row():
                btn_reset = gr.Button("Reset")
                btn_step = gr.Button("Step")

            with gr.Row():
                btn_up = gr.Button("Up")
                btn_down = gr.Button("Down")
                btn_left = gr.Button("Left")
                btn_right = gr.Button("Right")

            btn_reset.click(fn=ui_reset, outputs=[game_state, board_img, score_out, alive_out, dir_out])
            btn_step.click(fn=ui_step, inputs=[game_state], outputs=[game_state, board_img, score_out, alive_out, dir_out])

            btn_up.click(fn=lambda s: ui_turn(s, "up"), inputs=[game_state], outputs=[game_state, board_img, score_out, alive_out, dir_out])
            btn_down.click(fn=lambda s: ui_turn(s, "down"), inputs=[game_state], outputs=[game_state, board_img, score_out, alive_out, dir_out])
            btn_left.click(fn=lambda s: ui_turn(s, "left"), inputs=[game_state], outputs=[game_state, board_img, score_out, alive_out, dir_out])
            btn_right.click(fn=lambda s: ui_turn(s, "right"), inputs=[game_state], outputs=[game_state, board_img, score_out, alive_out, dir_out])

            gr.Markdown("Click **Reset** to initialize the board.")

        with gr.Tab("Snake (voice-controlled)"):
            game_state_v = gr.State(new_game(seed=0))

            board_img_v = gr.Image(label="Snake board", type="numpy")
            score_out_v = gr.Number(label="Score")
            alive_out_v = gr.Checkbox(label="Alive")
            dir_out_v = gr.Textbox(label="Direction")

            audio_in2 = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="Speak: up / down / left / right"
            )
            info_box = gr.Textbox(label="Prediction info")

            with gr.Row():
                btn_reset_v = gr.Button("Reset")
                btn_voice_step = gr.Button("Speak & Move")

            btn_reset_v.click(fn=ui_reset, outputs=[game_state_v, board_img_v, score_out_v, alive_out_v, dir_out_v])
            btn_voice_step.click(
                fn=voice_step_threshold,
                inputs=[game_state_v, audio_in2],
                outputs=[game_state_v, board_img_v, score_out_v, alive_out_v, dir_out_v, info_box],
            )

            gr.Markdown("Click **Reset** before using **Speak & Move**.")

if __name__ == "__main__":
    demo.launch()