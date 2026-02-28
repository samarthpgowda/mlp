import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, log_loss
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import shuffle

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -------------------------
# PATH
# -------------------------
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

# -------------------------
# PREPARE DATA
# -------------------------
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
y = np.where(y == "N", 0, 1)

X, y = shuffle(X, y, random_state=42)

# -------------------------
# NORMALIZATION
# -------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

np.save("scaler_mean.npy", scaler.mean_)
np.save("scaler_scale.npy", scaler.scale_)
print("Scaler parameters saved.")

# -------------------------
# SPLIT
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# MODEL
# -------------------------
model = MLPClassifier(
    hidden_layer_sizes=(16,),
    activation='relu',
    solver='adam',
    alpha=0.015,
    max_iter=1,
    warm_start=True,
    random_state=42
)

epochs = 150
train_loss = []
test_loss = []

for epoch in range(epochs):
    X_train, y_train = shuffle(X_train, y_train)
    model.fit(X_train, y_train)

    train_loss.append(model.loss_)
    y_test_prob = model.predict_proba(X_test)
    test_loss.append(log_loss(y_test, y_test_prob))

# -------------------------
# FLOATING EVALUATION
# -------------------------
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print("\n=== FLOATING POINT RESULTS ===")

print("\nTraining Accuracy:", accuracy_score(y_train, y_train_pred))
print("Training Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))

print("\nTest Accuracy:", accuracy_score(y_test, y_test_pred))
print("Test Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))

print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

# -------------------------
# PLOT LOSS
# -------------------------
plt.figure()
plt.plot(train_loss, label="Training Loss")
plt.plot(test_loss, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Training vs Test Loss")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# WEIGHT EXTRACTION
# -------------------------
W1 = model.coefs_[0]
W2 = model.coefs_[1]
b1 = model.intercepts_[0]
b2 = model.intercepts_[1]

all_params = np.concatenate([
    W1.flatten(),
    W2.flatten(),
    b1.flatten(),
    b2.flatten(),
    X.flatten()
])

max_abs = np.max(np.abs(all_params))
print("\nMaximum absolute value:", max_abs)

integer_bits = int(np.ceil(np.log2(max_abs + 1)))
total_bits = 16
fractional_bits = total_bits - 1 - integer_bits

print(f"Using Q{integer_bits}.{fractional_bits}")

scale_factor = 2 ** fractional_bits

W1_fixed = np.round(W1 * scale_factor).astype(np.int16)
W2_fixed = np.round(W2 * scale_factor).astype(np.int16)
b1_fixed = np.round(b1 * scale_factor).astype(np.int16)
b2_fixed = np.round(b2 * scale_factor).astype(np.int16)

np.savetxt("W1_fixed.csv", W1_fixed, fmt="%d", delimiter=",")
np.savetxt("W2_fixed.csv", W2_fixed, fmt="%d", delimiter=",")
np.savetxt("b1_fixed.csv", b1_fixed, fmt="%d", delimiter=",")
np.savetxt("b2_fixed.csv", b2_fixed, fmt="%d", delimiter=",")

np.save("fractional_bits.npy", fractional_bits)

print("Fixed-point weights saved.")
