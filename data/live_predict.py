import asyncio
from bleak import BleakClient
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis

ADDRESS   = "C0:CD:D6:85:67:1E"
CHAR_UUID = "abcd1234-5678-1234-5678-1234567890ab"

model  = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

WINDOW_SIZE    = 25
SMOOTHING      = 5
REQUIRED_COUNT = 200      # ~20 seconds at 10Hz
OUTPUT_FILE    = "live_output.txt"

# ── Motion gate: window must have enough movement to even be considered anxiety
# Tune this based on your calm data — calm roll_std is ~0.3–0.8, fidgeting is much higher
MOTION_THRESHOLD_PITCH_STD = 1.5   # pitch std dev must exceed this
MOTION_THRESHOLD_ROLL_STD  = 1.5   # roll std dev must exceed this
MOTION_THRESHOLD_VEL       = 0.5   # mean absolute velocity must exceed this

buffer          = []
last_preds      = []
anxiety_counter = 0


def is_fidgeting(data):
    """Return True only if the window shows real physical movement."""
    pitch = np.array([x[0] for x in data])
    roll  = np.array([x[1] for x in data])
    pitch_vel = np.abs(np.diff(pitch))
    roll_vel  = np.abs(np.diff(roll))
    
    pitch_std = np.std(pitch)
    roll_std  = np.std(roll)
    mean_vel  = (np.mean(pitch_vel) + np.mean(roll_vel)) / 2
    
    return (pitch_std > MOTION_THRESHOLD_PITCH_STD or 
            roll_std  > MOTION_THRESHOLD_ROLL_STD) and \
            mean_vel  > MOTION_THRESHOLD_VEL


def extract_features(data):
    pitch = np.array([x[0] for x in data])
    roll  = np.array([x[1] for x in data])

    pitch_vel = np.abs(np.diff(pitch))
    roll_vel  = np.abs(np.diff(roll))
    motion    = np.sqrt(pitch**2 + roll**2)

    return pd.DataFrame([{
        "pitch_mean":     np.mean(pitch),
        "pitch_std":      np.std(pitch),
        "pitch_range":    np.max(pitch) - np.min(pitch),
        "pitch_skew":     skew(pitch),
        "pitch_kurtosis": kurtosis(pitch),
        "roll_mean":      np.mean(roll),
        "roll_std":       np.std(roll),
        "roll_range":     np.max(roll) - np.min(roll),
        "roll_skew":      skew(roll),
        "roll_kurtosis":  kurtosis(roll),
        "motion_mean":    np.mean(motion),
        "motion_std":     np.std(motion),
        "pitch_vel_mean": np.mean(pitch_vel) if len(pitch_vel) > 0 else 0,
        "roll_vel_mean":  np.mean(roll_vel)  if len(roll_vel)  > 0 else 0,
        "total_velocity": np.mean(pitch_vel) + np.mean(roll_vel),
        "pitch_accel":    np.mean(np.abs(np.diff(pitch_vel))) if len(pitch_vel) > 1 else 0,
        "roll_accel":     np.mean(np.abs(np.diff(roll_vel)))  if len(roll_vel)  > 1 else 0,
        "pitch_zcr":      np.mean(np.diff(np.sign(pitch)) != 0),
        "roll_zcr":       np.mean(np.diff(np.sign(roll))  != 0),
    }])


def handle_data(sender, data):
    global buffer, last_preds, anxiety_counter

    try:
        p, r = map(float, data.decode().strip().split(','))
        buffer.append((p, r))

        if len(buffer) >= WINDOW_SIZE:
            window = buffer[-WINDOW_SIZE:]

            # ── Gate 1: no real movement → force CALM, reset counter
            if not is_fidgeting(window):
                anxiety_counter = 0
                last_preds.clear()
                status = "CALM"
            else:
                # ── Gate 2: movement detected → ask the model
                features   = extract_features(window)
                X          = scaler.transform(features)
                pred       = model.predict(X)[0]

                last_preds.append(pred)
                if len(last_preds) > SMOOTHING:
                    last_preds.pop(0)

                final_pred = max(set(last_preds), key=last_preds.count)

                if final_pred == 1:
                    anxiety_counter += 1
                else:
                    anxiety_counter = 0  # calm model output also resets

                status = "ANXIETY" if anxiety_counter >= REQUIRED_COUNT else "CALM"

            with open(OUTPUT_FILE, "w") as f:
                f.write(f"{status},{p},{r},{anxiety_counter}")

            print(f"[{status}]  pitch={p:.2f}  roll={r:.2f}  counter={anxiety_counter}")

    except Exception as e:
        print(f"[handle_data error] {e}")


async def main():
    async with BleakClient(ADDRESS) as client:
        print("Connected to ESP32")
        await client.start_notify(CHAR_UUID, handle_data)
        while True:
            await asyncio.sleep(1)


asyncio.run(main())