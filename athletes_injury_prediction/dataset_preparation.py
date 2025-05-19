import pandas as pd
import numpy as np

print("--- Starting Data Preparation ---")

# --- 1. Load Data ---
# Use dayfirst=True for daily_data and activity_data based on your provided format (DD-MM-YYYY)
try:
    athletes = pd.read_csv("athletes.csv")
    daily = pd.read_csv("daily_data.csv", parse_dates=["date"], dayfirst=True)
    activity = pd.read_csv("activity_data.csv", parse_dates=["date"], dayfirst=True)
    print("All datasets loaded successfully.\n")
except FileNotFoundError as e:
    print(f"Error: One of the CSV files not found. Please ensure 'athletes.csv', 'daily_data.csv', and 'activity_data.csv' are in the correct directory. Details: {e}")
    exit()

# Normalize athlete_id data types for consistent merging
athletes['athlete_id'] = athletes['athlete_id'].astype(str)
daily['athlete_id'] = daily['athlete_id'].astype(str)
activity['athlete_id'] = activity['athlete_id'].astype(str)
print("Athlete IDs normalized to string type.\n")

# --- 2. Drop Complex String Columns for Simplicity ---
activity.drop(columns=['hr_zones', 'power_zones'], inplace=True, errors='ignore')
athletes.drop(columns=['hr_zones', 'hrv_range'], inplace=True, errors='ignore') # Also drop hrv_range as it's complex
print("Complex HR/Power zone and HRV range columns dropped for simplicity.\n")


# --- 3. Aggregate Activity Data Daily ---
print("Aggregating activity data to daily summaries...\n")
activity_daily_aggregated = activity.groupby(['athlete_id', 'date'], as_index=False).agg(
    total_duration_minutes=('duration_minutes', 'sum'),
    total_tss=('tss', 'sum'),
    avg_daily_hr_activity=('avg_hr', 'mean'),
    max_daily_hr_activity=('max_hr', 'max'),
    total_distance_km=('distance_km', 'sum'),
    avg_daily_power=('avg_power', 'mean'),
    total_work_kilojoules=('work_kilojoules', 'sum'),
    total_elevation_gain=('elevation_gain', 'sum'),
    num_activities=('workout_type', 'count') # Total number of activities on that day
)

# Add daily counts for each sport (your excellent addition!)
activity_sport_dummies = pd.get_dummies(activity[['athlete_id', 'date', 'sport']], columns=['sport'], prefix='num_activities_sport')
activity_sport_counts = activity_sport_dummies.groupby(['athlete_id', 'date']).sum().reset_index()

# Merge aggregated activity data with sport counts
activity_daily = pd.merge(activity_daily_aggregated, activity_sport_counts, on=['athlete_id', 'date'], how='left')
print("Daily activity summary and sport counts created.\n")

# --- 4. Merge All DataFrames ---
print("Merging all DataFrames...\n")
combined = pd.merge(daily, activity_daily, on=['athlete_id', 'date'], how='left')

# --- DIAGNOSTIC STEP ---
# Before merging with athletes_df, let's see current columns in 'combined'
# and what columns are in 'athletes' to predict potential conflicts.
print("\n--- Diagnostic: Columns before final merge ---")
print("Columns in 'combined' before merging athletes:\n", combined.columns.tolist())
print("Columns in 'athletes' DataFrame:\n", athletes.columns.tolist())
print("-------------------------------------------\n")
# If 'resting_hr' is in both, it will become 'resting_hr_x' and 'resting_hr_y' after merge.


combined = pd.merge(combined, athletes, on='athlete_id', how='left')

# --- DIAGNOSTIC STEP AFTER MERGE ---
print("\n--- Diagnostic: Columns after final merge ---")
print("Columns in 'combined' after merging athletes:\n", combined.columns.tolist())
print("-------------------------------------------\n")


# Sort by athlete_id and date, crucial for time-series operations
combined.sort_values(by=['athlete_id', 'date'], inplace=True)
print("All data merged and sorted.\n")

# --- 5. Handle Missing Values ---
print("--- Handling Missing Values ---")
print("Missing values before handling:\n", combined.isnull().sum()[combined.isnull().sum() > 0])

# Fill activity-related NaNs (days with no training) with 0
# Find columns that came from activity_daily (excluding keys)
activity_related_cols = [col for col in activity_daily.columns if col not in ['athlete_id', 'date']]
combined[activity_related_cols] = combined[activity_related_cols].fillna(0)
print("\nActivity-related NaNs filled with 0.")

# Fill physiological/daily NaNs using forward-fill (per athlete). This assumes the last known value persists for each athlete.
# Based on your data, 'resting_hr' is in both daily and athletes.
# pandas will rename 'resting_hr' from 'daily' to 'resting_hr_x'and 'resting_hr' from 'athletes' to 'resting_hr_y' if both are present.
# We should generally use the daily physiological measurement ('resting_hr_x')and fill it based on its daily trend.
physiological_cols = [
    'resting_hr_x', # Corrected column name assuming it's from daily_data
    'hrv', 'sleep_hours', 'deep_sleep', 'light_sleep', 'rem_sleep',
    'sleep_quality', 'body_battery_morning', 'stress', 'body_battery_evening',
    'planned_tss', 'actual_tss'
]

# Filter physiological_cols to only include those actually present in `combined`
physiological_cols_present = [col for col in physiological_cols if col in combined.columns]

for col in physiological_cols_present:
    combined[col] = combined.groupby('athlete_id')[col].transform(lambda x: x.ffill())
print("Physiological data NaNs filled with forward-fill (per athlete).")

# Fill any remaining NaNs (e.g., at the very start of an athlete's record or in static athlete data)
# with column mean for numerical, or mode for categorical.
for col in combined.columns:
    if combined[col].isnull().any():
        if pd.api.types.is_numeric_dtype(combined[col]):
            combined[col] = combined[col].fillna(combined[col].mean())
        elif pd.api.types.is_object_dtype(combined[col]):
            combined[col] = combined[col].fillna(combined[col].mode()[0])
print("Any remaining NaNs (numerical/categorical) filled with column mean/mode.")
print("Missing values after handling:\n", combined.isnull().sum()[combined.isnull().sum() > 0])
if combined.isnull().sum().sum() == 0:
    print("All missing values handled.\n")

# --- 6. Feature Engineering (Key Time-Series Features) ---
print("--- Starting Key Feature Engineering ---")

# Calculate 7-day rolling average for resting_hr (a key physiological trend)
if 'resting_hr_x' in combined.columns:
    combined['7_day_avg_resting_hr'] = combined.groupby('athlete_id')['resting_hr_x'].rolling(window=7, min_periods=1).mean().reset_index(level=0, drop=True)
    print("7-day average resting HR created (using resting_hr_x).\n")
else:
    print("Warning: 'resting_hr_x' not found for 7-day average. Skipping this feature.\n")


# Acute to Chronic Workload Ratio (ACWR) - Critical for injury prediction
combined['7_day_tss'] = combined.groupby('athlete_id')['total_tss'].rolling(window=7, min_periods=1).sum().reset_index(level=0, drop=True)
combined['28_day_tss'] = combined.groupby('athlete_id')['total_tss'].rolling(window=28, min_periods=1).sum().reset_index(level=0, drop=True)

combined['acwr'] = combined['7_day_tss'] / combined['28_day_tss'].replace(0, np.nan)
combined['acwr'].fillna(0, inplace=True) # Fill where 28_day_tss was 0 with 0

# Handle potential inf values (if 7_day_tss > 0 and 28_day_tss == 0)
combined.loc[combined['acwr'] == np.inf, 'acwr'] = combined['acwr'].max() if not combined['acwr'].replace([np.inf, -np.inf], np.nan).isnull().all() else 0
print("Acute to Chronic Workload Ratio (ACWR) calculated.\n")

# --- 7. Encode Categorical Features ---
print("--- Encoding Categorical Features ---")
combined = pd.get_dummies(combined, columns=['gender', 'lifestyle'], drop_first=True)
print("Categorical features ('gender', 'lifestyle') one-hot encoded.\n")


# --- Final Check and Save ---
print("--- Final Data Snapshot ---")
print(combined.head())
print(f"\nFinal combined DataFrame shape: {combined.shape}")
print("Final combined DataFrame columns:\n", combined.columns.tolist())

# Save to CSV
combined.to_csv("merged_athlete_data.csv", index=False)
print("\n✅ Simplified merged data saved as 'merged_athlete_data.csv'")
