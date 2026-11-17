import os
import sys
import subprocess

# Auto-redirect to project virtual environment (.venv) if started with system Python
_venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe"))
if os.path.exists(_venv_python) and os.path.normcase(sys.executable) != os.path.normcase(_venv_python):
    sys.exit(subprocess.call([_venv_python] + sys.argv))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from app.recognition.dynamic.dynamic_classifier import DynamicClassifier

def evaluate(data_path="MP_Data", save_plot="confusion_matrix.png"):
    print("=" * 60)
    print(" Indian Sign Language (ISL) Model Evaluation & Confusion Matrix")
    print("=" * 60)

    # 1. Load active dynamic model
    print("[1/4] Loading model...")
    dc = DynamicClassifier()
    if dc.model is None:
        print("Error: Could not load active model. Please check models/ directory.")
        return

    classes = dc.classes
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    label_map = {label: i for i, label in enumerate(classes)}

    # 2. Load dataset
    print(f"\n[2/4] Loading dataset from '{data_path}'...")
    sequences = []
    labels = []

    for action in classes:
        act_dir = os.path.join(data_path, action)
        if not os.path.exists(act_dir):
            continue
        seq_dirs = [d for d in os.listdir(act_dir) if os.path.isdir(os.path.join(act_dir, d))]
        for seq in seq_dirs:
            window = []
            for frame_num in range(30):
                fpath = os.path.join(act_dir, seq, f"{frame_num}.npy")
                if os.path.exists(fpath):
                    window.append(np.load(fpath))
                else:
                    window.append(np.zeros(63))
            if len(window) == 30:
                sequences.append(window)
                labels.append(label_map[action])

    if not sequences:
        print(f"Error: No sequences found in {data_path}.")
        return

    X = np.array(sequences)
    y_true = np.array(labels)
    print(f"Loaded {len(X)} sequences across {len(np.unique(y_true))} classes.")

    # 3. Model predictions
    print("\n[3/4] Running model inference...")
    preds = dc.model.predict(X, verbose=0)
    y_pred = np.argmax(preds, axis=1)

    acc = np.mean(y_true == y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    # 4. Print results
    print("\n" + "=" * 60)
    print(f" OVERALL ACCURACY: {acc * 100:.2f}% ({np.sum(y_true == y_pred)}/{len(y_true)} correct)")
    print("=" * 60)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    print("\n--- Confusion Matrix (Rows = True Class, Columns = Predicted Class) ---")
    # Header
    col_width = max(len(c) for c in classes) + 2
    header = f"{'True \\ Pred':<{col_width}}" + "".join(f"{c:>{col_width}}" for c in classes)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(cm):
        row_str = f"{classes[i]:<{col_width}}" + "".join(f"{val:>{col_width}}" for val in row)
        print(row_str)

    # 5. Save visual heatmap plot
    print(f"\n[4/4] Generating visual confusion matrix plot to '{save_plot}'...")
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel='True Label',
        xlabel='Predicted Label',
        title=f'ISL Recognition Confusion Matrix (Accuracy: {acc*100:.2f}%)'
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_plot, dpi=300)
    plt.close()
    print(f"Confusion matrix plot successfully saved to: {os.path.abspath(save_plot)}")
    print("=" * 60)

if __name__ == '__main__':
    evaluate()

