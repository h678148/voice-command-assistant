# Voice Command Assistant

A deep learning project for **voice command recognition** and **interactive control**, developed as part of the **DAT255 – Deep Learning Engineering** course.

The system recognizes simple spoken commands (*up, down, left, right*) and demonstrates real-time inference by controlling a **Snake game with voice commands**.

---

## Deployed application

Try the application here:

**https://bilals04-voice-command-assistant.hf.space**

Includes:

- Voice command classifier
- Snake game with manual controls
- Snake game controlled by voice commands
- Live voice controlled snake game 

---

## YouTube Demo  

A short demonstration of the application:

**https://www.youtube.com/watch?v=AKgTlB2krDQ**

---

## Project Overview

This project follows a standard **machine learning lifecycle**:

1. Data ingestion and verification  
2. Exploratory data analysis and preprocessing  
3. Model training (CNN trained from scratch)  
4. Model evaluation and error analysis  
5. Experimentation and model improvements  
6. Deployment as an interactive web application  

The final system uses a **Convolutional Neural Network trained on the Google Speech Commands dataset**.

---

## Repository Structure
```
📦 voice-command-assistant
├─ notebooks/       # Development notebooks (ML lifecycle)
├─ models/          # Trained model checkpoints
├─ results/         # Metrics, figures, experiment outputs
├─ report/          # Project report
├─ deployment/      # Files used for the web application
└─ data/            # Data structure (dataset not included due to size)
```

---

## Technologies Used

- Python  
- PyTorch  
- Librosa  
- Gradio  
- Hugging Face Spaces  
- Google Colab  

---

## Dataset

The project uses the **Google Speech Commands Dataset**:

https://huggingface.co/datasets/google/speech_commands

