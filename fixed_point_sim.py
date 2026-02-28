import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle

DATA_FOLDER = "extracted_beats"

# -------------------------
# LOAD DATA
# -------------------------
all_data = []
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(DATA_FOLDER, file))
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)
print("Total samples:", len(data))

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = np.where(y == "N", 0, 1)

# Load scaler
mean = np.load("scaler_mean.npy")
scale = np.load("scaler_scale.npy")

X = (X - mean) / scale

X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load fixed weights
W1_fixed = np.loadtxt("W1_fixed.csv", delimiter=",").astype(np.int16)
W2_fixed = np.loadtxt("W2_fixed.csv", delimiter=",").astype(np.int16)
b1_fixed = np.loadtxt("b1_fixed.csv", delimiter=",").astype(np.int16)
b2_fixed = np.loadtxt("b2_fixed.csv", delimiter=",").astype(np.int16)

fractional_bits = int(np.load("fractional_bits.npy"))
scale_factor = 2 ** fractional_bits

X_train_fixed = np.round(X_train * scale_factor).astype(np.int16)
X_test_fixed = np.round(X_test * scale_factor).astype(np.int16)

input_size = W1_fixed.shape[0]
hidden_size = W1_fixed.shape[1]

def fixed_point_inference(x):

    hidden = np.zeros(hidden_size, dtype=np.int16)

    for j in range(hidden_size):
        acc = np.int64(0)
        for i in range(input_size):
            product = (np.int64(x[i]) * np.int64(W1_fixed[i][j])) >> fractional_bits
            acc += product

        acc += np.int64(b1_fixed[j])

        if acc < 0:
            acc = 0

        hidden[j] = np.int16(acc)

    acc_out = np.int64(0)
    for j in range(hidden_size):
        product = (np.int64(hidden[j]) * np.int64(W2_fixed[j])) >> fractional_bits
        acc_out += product

    acc_out += np.int64(b2_fixed)

    return 1 if acc_out > 0 else 0

# Train inference
y_train_pred_fixed = [fixed_point_inference(x) for x in X_train_fixed]

# Test inference
y_test_pred_fixed = [fixed_point_inference(x) for x in X_test_fixed]

print("\n=== FIXED POINT RESULTS ===")

print("\nTraining Accuracy:", accuracy_score(y_train, y_train_pred_fixed))
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred_fixed))

print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred_fixed))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred_fixed))

print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred_fixed))
