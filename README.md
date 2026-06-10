# Anxiety Detection System using ESP32 and Machine Learning

## Overview

This project is a real-time Anxiety Detection System that combines wearable motion sensing with Machine Learning to identify anxiety-related movement patterns.

The system uses an ESP32 microcontroller connected to motion sensors to collect user movement data. Features are extracted from sensor readings and processed using a trained Random Forest classifier to predict anxiety states in real time.

---

## Features

* Real-time motion data acquisition
* ESP32-based sensor platform
* Accelerometer and Gyroscope integration
* Sliding window feature extraction
* Machine Learning-based anxiety classification
* Live prediction and monitoring
* Scalable architecture for future wearable integration

---

## Hardware Components

* ESP32 Development Board
* Accelerometer Sensor
* Gyroscope Sensor

---

## Machine Learning Pipeline

### Data Collection

Sensor readings are collected continuously and stored for processing.

### Feature Extraction

The following features are extracted from each data window:

* Mean
* Standard Deviation
* Range
* Skewness
* Kurtosis
* Velocity Features
* Acceleration Features
* Zero Crossing Rate (ZCR)

### Data Preprocessing

* StandardScaler normalization

### Classification Model

* Random Forest Classifier
* 400 Decision Trees
* Balanced Class Weighting
* Maximum Depth = 15

---

## Project Structure

```text
ANXIETY_DETECTION
│
├── arduino_code/
├── backend/
├── frontend/
├── models/
│
├── data/
│   ├── data.csv
│   ├── model.pkl
│   ├── scaler.pkl
│
├── collect_data.py
├── train_model.py
├── live_predict.py
├── app.py
└── README.md
```

## Installation

```bash
git clone https://github.com/RiyaDShetty/Anxiety-detection-system.git

cd Anxiety-detection-system

pip install -r requirements.txt
```

## Running the Project

### Train the Model

```bash
python train_model.py
```

### Run Live Prediction

```bash
python live_predict.py
```

### Launch Application

```bash
python app.py
```

## Future Improvements

* Integration with wearable devices
* Cloud-based analytics
* Mobile application support
* Deep Learning models for improved accuracy
* Real-time healthcare monitoring dashboard

Riya D Shetty
Software Engineering Student
