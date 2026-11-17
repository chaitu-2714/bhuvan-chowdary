# Indian Sign Language (ISL) Recognition System

A real-time Indian Sign Language (ISL) recognition desktop system powered by OpenCV, MediaPipe, TensorFlow / Keras (LSTM), Scikit-Learn, and PySide6 (Qt).

---

## 🌟 Features

- **Real-Time Hand Landmark Detection:** Uses Google MediaPipe to capture and track 21 hand landmarks (dual hand support).
- **Dual Recognition Engine:**
  - **Static Alphabet Classification:** Fast classification for static gestures and letters using SVM, Random Forest, and MLP classifiers.
  - **Dynamic Sign Recognition:** Temporal sequence prediction using multi-layer LSTM neural networks (30-frame sequence window).
- **Modern Desktop GUI (PySide6):**
  - Live video feed with landmark visualization.
  - Interactive dashboard with telemetry (CPU, RAM, FPS).
  - Built-in Dataset Collector to record static gestures and dynamic sequences.
  - Model Training console with real-time progress and confusion matrix logging.
- **Multilingual Translation & Text-to-Speech:**
  - Translates recognized gestures into multiple languages (English, Hindi, Telugu).
  - Offline text-to-speech output using `pyttsx3`.
- **Model Evaluation:**
  - Precision, Recall, F1-score classification report.
  - Automated visual confusion matrix generation (`evaluate_model.py`).

---

## 📁 Project Structure

```
├── app/
│   ├── camera/             # Camera management and frame capture
│   ├── database/           # SQLite persistence for sessions and histories
│   ├── dataset/            # Dataset recording and batch management
│   ├── evaluation/         # Performance metrics & confusion matrix
│   ├── features/           # Hand landmark keypoint normalization
│   ├── landmarks/          # MediaPipe Hands detector
│   ├── models/             # Model manager and registry
│   ├── monitoring/         # Telemetry and logging
│   ├── recognition/        # Static & dynamic sign classification engines
│   ├── speech/             # Text-to-Speech (pyttsx3) worker thread
│   ├── translation/        # Multilingual translation service
│   └── ui/                 # PySide6 Desktop GUI and background QThreads
├── MP_Data/                # Dynamic sign dataset sequence recordings
├── data/                   # Dynamic and static alphabet datasets
├── models/                 # Saved models (static PKL, dynamic H5)
├── tests/                  # Unit test suite
├── desktop_app.py          # Main PySide6 Desktop Application launcher
├── evaluate_model.py       # Model evaluation & confusion matrix generator
├── inference.py            # Standalone OpenCV real-time prediction
├── collect_data.py         # Standalone data collection script
├── train_model.py          # Standalone LSTM training script
└── requirements.txt        # Project dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.12 (Recommended for MediaPipe compatibility)

### 2. Installation
Clone the repository:
```bash
git clone https://github.com/chaitu-2714/bhuvan-chowdary.git
cd bhuvan-chowdary
```

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

---

## 🖥️ Usage

### Run the Main Desktop Application
```bash
python app/ui/desktop_app.py
```

### Run Model Evaluation & Confusion Matrix
```bash
python evaluate_model.py
```

### Run Lightweight Standalone Inference
```bash
python inference.py
```

### Run Unit Tests
```bash
python -m unittest tests/test_isl_system.py
```

---

## 📊 Dataset Classes (Dynamic)

- `hello`
- `thanks`
- `iloveyou`
- `thumbs_up`
- `thumbs_down`
- `peace`
- `ok`
- `stop`
- `point_up`
