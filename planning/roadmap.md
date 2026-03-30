# Voice-Command CLI: Project Roadmap

Detailed list of items to accomplish for the CS50 final project.

## Phase 1: Environment & Setup
- [ ] **Create a Virtual Environment**: Keep dependencies isolated (`python -m venv venv`).
- [ ] **Install Core Libraries**:
  ```bash
  pip install librosa scikit-learn sounddevice numpy soundfile
  ```
- [ ] **Verify Audio Input**: Run a simple `sounddevice` script to ensure the microphone is recognized by Python.

## Phase 2: Data Collection (The "Dataset" Script)
- [ ] **Define Your Vocabulary**: Choose 5–10 distinct words (e.g., "Open", "Close", "Status", "Code", "Exit").
- [ ] **Build a Recorder Script**: Create a script that:
  - Prompts for a word.
  - Records for exactly 1.5 or 2 seconds.
  - Saves the file as a `.wav` in a folder named after the word (e.g., `data/open/01.wav`).
- [ ] **Record Samples**: Aim for at least 20–30 recordings per word for sufficient variety.

## Phase 3: Feature Extraction (The "Brain" Prep)
- [ ] **MFCC Extraction**: Write a function using `librosa.feature.mfcc` to convert a `.wav` file into a 2D array of coefficients.
- [ ] **Data Flattening**: Since `scikit-learn` expects a 1D input per sample, average the MFCCs over time or flatten the array.
- [ ] **Preprocessing Script**: Create a script that loops through `data/` folders, extracts features, and saves them into `X` (features) and `y` (labels) files (using `numpy.save` or CSV).

## Phase 4: Model Training
- [ ] **Split Data**: Use `train_test_split` from `sklearn` to set aside 20% of data for testing.
- [ ] **Choose a Classifier**: Start with **K-Nearest Neighbors (KNN)** or a **Support Vector Machine (SVM)**.
- [ ] **Evaluation**: Verify accuracy (aim for >90%).
- [ ] **Save the Model**: Use `joblib` or `pickle` to save the trained model.

## Phase 5: The Live CLI (Integration)
- [ ] **Audio Listener**: Create a loop that stays "active" and listens for a threshold of sound.
- [ ] **Live Prediction**: When sound is detected, capture 2 seconds, extract MFCCs, and pass them to the saved model.
- [ ] **Command Mapping**: Create a dictionary mapping words to shell commands (e.g., `{"open": "open ."}`).
- [ ] **Execution**: Use `subprocess` to run the command if the model is confident.

## Phase 6: Refinement
- [ ] **Noise Handling**: Add a "background noise" category to training data to reduce false positives.
- [ ] **Feedback**: Add terminal output or sound effects to signal successful recognition.
