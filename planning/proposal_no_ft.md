# Final Project Proposal: Voice-Command CLI (High-Level)

## Goal
Create a command-line interface (CLI) that recognizes a small set of spoken words (5–10) and executes corresponding system commands.

## Tech Stack
- **`librosa`**: For audio feature extraction (MFCCs).
- **`scikit-learn`**: For a simple classifier (e.g., K-Nearest Neighbors or SVM).
- **`sounddevice`**: For real-time audio recording.
- **`numpy`**: For basic data manipulation.

## Core Features
1. **Data Collection**: A script to record and label short (1-2s) audio clips of 5-10 distinct words (e.g., "open", "close", "start", "stop").
2. **Feature Extraction**: Use `librosa.feature.mfcc` to convert raw audio into a compact set of Mel-frequency cepstral coefficients.
3. **Training**: Train a `scikit-learn` model on the collected MFCC features.
4. **Recognition**: A live loop that listens for audio, extracts features, predicts the word, and executes a mapped command (e.g., "open" runs `open .` in the terminal).

## Why this is manageable
By using `librosa`, you avoid the complex math of signal processing and focus on the application logic: data handling, model training, and CLI integration.
