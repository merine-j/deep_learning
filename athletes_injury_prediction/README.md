# 🏃‍♂️ Athlete Injury Prediction with Soft Voting Ensemble using LSTM, GRU

This project predicts potential injuries in athletes using physiological and activity-based time-series data. It leverages deep learning models (LSTM and GRU) in a soft voting ensemble to improve prediction accuracy.

---

## 📁 Project Structure

📂 athlete-injury-prediction

├── dataset_preparation.py  # Merges and processes raw datasets

├── lstm.py  # LSTM model for injury prediction

├── gru.py  # GRU model for injury prediction

├── softvoting_ensemble.py  # Combines LSTM and GRU using soft voting


---

## 📦 Dataset Setup


1. **Download the raw datasets** from Zenodo:  
   🔗 [https://zenodo.org/records/15401061](https://zenodo.org/records/15401061)

2. Ensure the following files are in the project root:
   - `athletes.csv`
   - `daily_data.csv`
   - `activity_data.csv`

3. Run the script:
   ```bash
   dataset_preparation.py

This will generate a cleaned and feature-enriched merged_athlete_data.csv

🚀 Models

📌 lstm.py

Uses LSTM to predict injuries from time-series data.

Includes full model training, evaluation (accuracy, precision, recall, F1, AUC), and plots (loss, confusion matrix, ROC).

Saves the model as athlete_injury_lstm_model.h5.

📌 gru.py

Similar to the LSTM model but uses a GRU-based architecture.

Outputs same metrics and visualizations.

Saves the model as athlete_injury_gru_model.h5.

📌 softvoting_ensemble.py

Loads both trained models.

Averages predictions from LSTM and GRU (soft voting).

Evaluates ensemble predictions and plots ROC, PR curve, and confusion matrix.


