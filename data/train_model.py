import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# ===== LOAD DATA =====
df = pd.read_csv("data.csv", header=None)
df.columns = ["pitch", "roll", "label"]

# ===== WINDOW FEATURES =====
window_size = 25
features = []

for i in range(len(df) - window_size):
    window = df.iloc[i:i+window_size]

    pitch = window["pitch"].values
    roll = window["roll"].values

    pitch_vel = np.abs(np.diff(pitch))
    roll_vel = np.abs(np.diff(roll))
    motion = np.sqrt(pitch**2 + roll**2)

    feat = {
        "pitch_mean": np.mean(pitch),
        "pitch_std": np.std(pitch),
        "pitch_range": np.max(pitch) - np.min(pitch),
        "pitch_skew": skew(pitch),
        "pitch_kurtosis": kurtosis(pitch),

        "roll_mean": np.mean(roll),
        "roll_std": np.std(roll),
        "roll_range": np.max(roll) - np.min(roll),
        "roll_skew": skew(roll),
        "roll_kurtosis": kurtosis(roll),

        "motion_mean": np.mean(motion),
        "motion_std": np.std(motion),

        "pitch_vel_mean": np.mean(pitch_vel) if len(pitch_vel) > 0 else 0,
        "roll_vel_mean": np.mean(roll_vel) if len(roll_vel) > 0 else 0,
        "total_velocity": np.mean(pitch_vel) + np.mean(roll_vel),

        "pitch_accel": np.mean(np.abs(np.diff(pitch_vel))) if len(pitch_vel) > 1 else 0,
        "roll_accel": np.mean(np.abs(np.diff(roll_vel))) if len(roll_vel) > 1 else 0,

        "pitch_zcr": np.mean(np.diff(np.sign(pitch)) != 0),
        "roll_zcr": np.mean(np.diff(np.sign(roll)) != 0),

        "label": window["label"].iloc[0]
    }

    features.append(feat)

df_feat = pd.DataFrame(features)

# ===== SPLIT =====
X = df_feat.drop("label", axis=1)
y = df_feat["label"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# ===== MODEL =====
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    min_samples_split=3,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ===== EVALUATE =====
y_pred = model.predict(X_test)

print("\n===== RESULTS =====")
print("Accuracy:", model.score(X_test, y_test))
print("\n", classification_report(y_test, y_pred))

# ===== SAVE =====
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved!")