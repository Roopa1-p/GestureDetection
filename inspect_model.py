from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

# Step 1: Load the saved model
model = load_model("gesture_lstm_model.h5")

# Step 2: Print model summary in terminal
print("✅ Model Architecture:")
model.summary()

# Step 3: Save architecture to a JSON file (human-readable)
model_json = model.to_json()
with open("gesture_lstm_model.json", "w") as json_file:
    json_file.write(model_json)
print("✅ Model architecture saved as gesture_lstm_model.json")

# Step 4: Save a visual diagram of the model
plot_model(model, to_file="gesture_lstm_model.png", show_shapes=True, show_layer_names=True)
print("✅ Model diagram saved as gesture_lstm_model.png")
