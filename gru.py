import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load Merged Data
df = pd.read_csv("merged_athlete_data.csv", parse_dates=["date"])

# 2. Sort & Fill Remaining NaNs
df.sort_values(by=['athlete_id', 'date'], inplace=True)
df.fillna(method='ffill', inplace=True)
df.fillna(0, inplace=True)

# 3. Define Target and Features
target_col = 'injury'  # must be a binary column (0 = no injury, 1 = injury)
if target_col not in df.columns:
    raise ValueError(f"Column '{target_col}' not found in dataset. Please check.")

drop_cols = ['athlete_id', 'date']
X = df.drop(columns=drop_cols + [target_col])
y = df[target_col]

# 4. Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Reshape for GRU: (samples, time_steps, features)
def create_sequences(data, labels, seq_length=14):
    X_seq, y_seq = [], []
    athletes = df['athlete_id'].unique()
    for athlete in athletes:
        athlete_data = data[df['athlete_id'] == athlete]
        athlete_labels = labels[df['athlete_id'] == athlete]
        for i in range(len(athlete_data) - seq_length):
            X_seq.append(athlete_data.iloc[i:i+seq_length].values)
            y_seq.append(athlete_labels.iloc[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(pd.DataFrame(X_scaled), y, seq_length=14)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

# 7. Build GRU Model
model = Sequential([
    GRU(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 8. Train Model
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, callbacks=[early_stop], verbose=1)

# 9. Evaluate Model
y_pred_prob = model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"\n✅ Evaluation Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

# 10. Plot: Loss Curve
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("GRU Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Binary Crossentropy Loss")
plt.legend()
plt.grid(True)
plt.show()

# 11. Plot: Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("GRU Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 12. Plot: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.title("GRU ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# 13. Predict on New Sample
new_sample = np.expand_dims(X_test[0], axis=0)
new_prediction = model.predict(new_sample)
predicted_class = (new_prediction > 0.5).astype(int)
print("GRU Injury risk prediction:", "Injured" if predicted_class[0][0] == 1 else "Not Injured")

# 14. Save GRU Model
model.save("athlete_injury_gru_model.h5")
print("✅ GRU model saved as 'athlete_injury_gru_model.h5'")
