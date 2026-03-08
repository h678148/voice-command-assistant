# Voice command assistant (DAT255)

A low-latency voice command recognition system trained on the Google Speech Commands dataset.

## Project goals
- Train at least one deep learning model from scratch to recognize a small set of commands (e.g., up/down/left/right).
- Compare architectures and preprocessing strategies.
- Evaluate on unseen data with clear metrics and error analysis.
- (Optional) Deploy as a small web app and/or use microphone input to control a simple game (Snake).

## Dataset
Google Speech Commands (Hugging Face):
https://huggingface.co/datasets/google/speech_commands

## Repo structure
- `notebooks/`: End-to-end ML lifecycle
- `src/`: Reusable code for data, models, training and evaluation
- `app/`: Gradio/Streamlit demo apps
- `reports/`: Report and figures
