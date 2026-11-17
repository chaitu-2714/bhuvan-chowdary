import os
import sys
import subprocess

# Auto-redirect to project virtual environment (.venv) if started with system Python
_venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe"))
if os.path.exists(_venv_python) and os.path.normcase(sys.executable) != os.path.normcase(_venv_python):
    sys.exit(subprocess.call([_venv_python] + sys.argv))

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['hello', 'thanks', 'iloveyou', 'thumbs_up', 'thumbs_down', 'peace', 'ok', 'stop', 'point_up'])

# Thirty videos worth of data
no_sequences = 15

# Videos are going to be 30 frames in length
sequence_length = 30

# 1. Prepare Data
label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            try:
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            except FileNotFoundError:
                print(f"Warning: Missing data for {action} sequence {sequence} frame {frame_num}. Using zeros.")
                window.append(np.zeros(63))
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 2. Build and Train LSTM Neural Network
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("Starting training...")
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])
print("Training complete.")

# 3. Save weights
model.save('action.h5')
print("Model saved to action.h5")
