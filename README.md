# Real-Time Gesture Detection using Webcam

An experimental hand gesture recognition system using [MediaPipe](https://mediapipe.dev/) and an LSTM neural network.  
It can detect a few predefined gestures from a webcam feed in real time and trigger actions (e.g., scrolling, moving the mouse, printing messages).

## Features

- Collects training data using MediaPipe hand landmarks
- Trains an LSTM model to classify gestures
- Performs real-time gesture recognition via webcam
- Optional desktop automation using PyAutoGUI

## Requirements

- Python 3.11 (recommended)
- TensorFlow 2.x
- OpenCV (opencv-python or opencv-python-headless)
- MediaPipe
- NumPy
- scikit-learn
- (Optional) PyAutoGUI for controlling the mouse/keyboard

Install dependencies (inside a virtual environment):

```bash
pip install tensorflow opencv-python mediapipe scikit-learn pyautogui


