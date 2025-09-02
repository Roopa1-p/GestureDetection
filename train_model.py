import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

GESTURES = ['open_hand', 'thumbs_up', 'super']
DATA_PATH = 'MP_Data'
SEQUENCE_LENGTH = 30

label_map = {label: idx for idx, label in enumerate(GESTURES)}
sequences, labels = [], []

for gesture in GESTURES:
    for sequence in range(30):
        window = []
        for frame_num in range(SEQUENCE_LENGTH):
            path = os.path.join(DATA_PATH, gesture, str(sequence), f"{frame_num}.npy")
            if os.path.isfile(path):
                window.append(np.load(path))
        if len(window) == SEQUENCE_LENGTH:
            sequences.append(window)
            labels.append(label_map[gesture])

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 126)))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(GESTURES), activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

model.save('gesture_lstm_model.h5')
print("✅ Model trained and saved as 'gesture_lstm_model.h5'")
