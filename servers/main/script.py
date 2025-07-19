import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_gesture_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "label_encoder_classes.npy")

print(f"📁 MODEL_PATH: {MODEL_PATH}")
print(f"📁 LABELS_PATH: {LABELS_PATH}")
print("🧪 ¿Existe modelo?:", os.path.exists(MODEL_PATH))
print("🧪 ¿Existe labels?:", os.path.exists(LABELS_PATH))
