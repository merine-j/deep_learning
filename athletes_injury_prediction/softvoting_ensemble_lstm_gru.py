import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from tensorflow.keras.models import load_model

# 1. Load Data
df = pd.read_csv("merged_athlete_data.csv", parse_dates=["date"])
df.sort_values(by=["athlete_id", "date"], inplace=True)
df.fillna(method="ffill", inplace=True)
df.fillna(0, inplace=True)

# 2. Feature Selection and Scaling
target_col = "injury"
drop_cols = ["athlete_id", "date"]
X = df.drop(columns=drop_cols + [target_col])
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Sequence Preparation (same as before)
def create_sequences(data, labels, seq_length=14):
    X_seq, y_seq = [], []
    athletes = df["athlete_id"].unique()
    for athlete in athletes:
        athlete_data = data[df["athlete_id"] == athlete]
        athlete_labels = labels[df["athlete_id"] == athlete]
        for i in range(len(athlete_data) - seq_length):
            X_seq.append(athlete_data.iloc[i:i+seq_length].values)
            y_seq.append(athlete_labels.iloc[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(pd.DataFrame(X_scaled), y, seq_length=14)

# 4. Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42, stratify=y_seq)

# 5. Load Saved Models
model_lstm = load_model("athlete_injury_lstm_model.h5")
model_gru = load_model("athlete_injury_gru_model.h5")

# 6. Soft Voting: Average Predictions
y_pred_lstm_prob = model_lstm.predict(X_test).ravel()
y_pred_gru_prob = model_gru.predict(X_test).ravel()
y_pred_ensemble_prob = (y_pred_lstm_prob + y_pred_gru_prob) / 2
y_pred_ensemble = (y_pred_ensemble_prob > 0.5).astype(int)

# 7. Evaluation
acc = accuracy_score(y_test, y_pred_ensemble)
prec = precision_score(y_test, y_pred_ensemble)
rec = recall_score(y_test, y_pred_ensemble)
f1 = f1_score(y_test, y_pred_ensemble)
auc = roc_auc_score(y_test, y_pred_ensemble_prob)

print("\n✅ Soft Voting Ensemble Evaluation:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

# 1. Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred_ensemble)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Soft Voting Ensemble)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_ensemble_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve (Soft Voting Ensemble)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()

# 3. Precision-Recall Curve
precisions, recalls, _ = precision_recall_curve(y_test, y_pred_ensemble_prob)
plt.figure(figsize=(6, 5))
plt.plot(recalls, precisions, color='purple')
plt.title("Precision-Recall Curve (Soft Voting Ensemble)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.show()

# 4. Optional: Prediction Probability Distribution
plt.figure(figsize=(6, 4))
sns.histplot(y_pred_ensemble_prob, bins=50, kde=True, color='teal')
plt.title("Prediction Probability Distribution")
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
