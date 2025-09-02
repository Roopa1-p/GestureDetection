import os
import sys
import cv2
import time
import numpy as np
import pyautogui
import contextlib
from collections import deque
from tensorflow.keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull), contextlib.redirect_stdout(fnull):
            yield

with suppress_stdout_stderr():
    import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

model = load_model('gesture_lstm_model.h5')
GESTURES = ['open_hand', 'thumbs_up', 'super']
SEQUENCE_LENGTH = 30

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam.")
    sys.exit()
else:
    print("✅ Webcam opened successfully.")

sequence = deque(maxlen=SEQUENCE_LENGTH)
predictions = deque(maxlen=10)
pyautogui.FAILSAFE = False
prev_action_time = time.time()

def perform_action(gesture):
    global prev_action_time
    now = time.time()
    if now - prev_action_time < 0.5:
        return
    if gesture == 'open_hand':
        pyautogui.moveRel(20, 0)
    elif gesture == 'thumbs_up':
        pyautogui.scroll(100)
    elif gesture == 'super':
        print("🎉 Super gesture detected!")
    prev_action_time = now

def extract_keypoints(results):
    lh = np.zeros(21 * 3)
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    rh = np.zeros(21 * 3)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    return np.concatenate([lh, rh])

with suppress_stdout_stderr():
    with mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame from webcam.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

            if len(sequence) == SEQUENCE_LENGTH:
                input_seq = np.expand_dims(np.array(sequence), axis=0)  # Shape: (1, 30, 126)
                print(f"Input shape for prediction: {input_seq.shape}")  # Debug line (optional)
                res = model.predict(input_seq, verbose=0)[0]
                predicted_gesture = GESTURES[np.argmax(res)]
                confidence = res[np.argmax(res)]
                predictions.append(predicted_gesture)

                if predictions.count(predicted_gesture) > 2:
                    perform_action(predicted_gesture)
                    cv2.putText(image, f'Gesture: {predicted_gesture} ({confidence:.2f})',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Gesture Mouse Control", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                print("ℹ️ Quitting...")
                break

cap.release()
cv2.destroyAllWindows()
