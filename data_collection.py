import cv2
import numpy as np
import os
import mediapipe as mp
import time

actions = ['open_hand', 'thumbs_up', 'super']
no_sequences = 30
sequence_length = 30
start_folder = 0
DATA_PATH = os.path.join('MP_Data')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    lh = np.zeros(21*3)
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    rh = np.zeros(21*3)
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    return np.concatenate([lh, rh])

def main():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
        for action in actions:
            print(f"Collecting data for {action}")
            time.sleep(2)
            for sequence in range(start_folder, start_folder + no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    cv2.putText(image, f'{action} | Video {sequence} | Frame {frame_num}',
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                    np.save(npy_path, keypoints)

                    cv2.imshow('Collecting Gestures', image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
