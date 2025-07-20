import cv2
import numpy as np
import os
from collections import deque
import mediapipe as mp
import requests
from flask import Flask, render_template, Response, jsonify

# === Manejo de TensorFlow/Keras ===
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    try:
        import keras
        from keras.models import load_model
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
        print("⚠️ TensorFlow/Keras no disponible - solo modo demo")

app = Flask(__name__)

# === Cargar modelo y etiquetas ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_gesture_model.h5")
MODEL_DEMO_PATH = os.path.join(BASE_DIR, "hand_gesture_model_demo.h5") 
LABELS_PATH = os.path.join(BASE_DIR, "label_encoder_classes.npy")

model = None
label_classes = []

if TF_AVAILABLE:
    try:
        if os.path.exists(LABELS_PATH):
            label_classes = np.load(LABELS_PATH, allow_pickle=True)
            print(f"✅ Labels cargados: {len(label_classes)} gestos")
        else:
            label_classes = ["Rock", "Paper", "Scissors", "Peace", "Fist"]
            
        model_loaded = False
        
        if os.path.exists(MODEL_PATH):
            try:
                with tf.keras.utils.custom_object_scope({'DTypePolicy': None}):
                    model = load_model(MODEL_PATH, compile=False)
                    print("✅ Modelo original cargado exitosamente")
                    model_loaded = True
            except Exception as e1:
                print(f"❌ Error cargando modelo original: {e1}")
                if os.path.exists(MODEL_DEMO_PATH):
                    try:
                        model = load_model(MODEL_DEMO_PATH, compile=False)
                        print("✅ Modelo demo cargado exitosamente")
                        model_loaded = True
                    except Exception as e2:
                        print(f"❌ Error cargando modelo demo: {e2}")
        
        if not model_loaded:
            print("⚠️ Ningún modelo pudo cargarse. Modo demo activado.")
            model = None
            
    except Exception as e:
        print(f"❌ Error general cargando modelo: {e}")
        model = None
        try:
            label_classes = np.load(LABELS_PATH, allow_pickle=True)
            print(f"✅ Labels reales cargados: {len(label_classes)} gestos")
        except:
            label_classes = ["Rock", "Paper", "Scissors", "Peace", "Fist"]
else:
    print("⚠️ TensorFlow no disponible - Modo demo activado")
    model = None
    try:
        label_classes = np.load(LABELS_PATH, allow_pickle=True)
        print(f"✅ Labels reales cargados: {len(label_classes)} gestos")
    except:
        label_classes = ["Rock", "Paper", "Scissors", "Peace", "Fist"]

# === Configuración ===
SEQUENCE_LENGTH = 50
NUM_FEATURES = 126
sequence = deque(maxlen=SEQUENCE_LENGTH)

# === Inicializar MediaPipe ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# === Variables globales ===
cap = None
camera_active = False
current_gesture = ""
current_confidence = 0.0

def init_camera():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows compatibility
    return cap is not None and cap.isOpened()

def generate_frames():
    global camera_active, current_gesture, current_confidence

    while camera_active:
        if cap is None or not cap.isOpened():
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hand_landmarks = []
        if results.multi_hand_landmarks:
            for hand in results.multi_hand_landmarks:
                for lm in hand.landmark:
                    hand_landmarks.extend([lm.x, lm.y, lm.z])
            if len(results.multi_hand_landmarks) == 1:
                hand_landmarks.extend([0] * 63)

            if len(hand_landmarks) == NUM_FEATURES:
                sequence.append(hand_landmarks)
                for hand in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        if len(sequence) == SEQUENCE_LENGTH:
            if model is not None:
                input_data = np.expand_dims(sequence, axis=0)
                pred_probs = model.predict(input_data, verbose=0)
                pred_index = np.argmax(pred_probs)
                pred_label = label_classes[pred_index]
                confidence = pred_probs[0][pred_index]
            else:
                import random
                pred_label = random.choice(label_classes)
                confidence = random.uniform(0.7, 0.95)

            current_gesture = pred_label
            current_confidence = confidence

            # Enviar al servidor
            try:
                response = requests.post("http://localhost:5009/generate", json={"prompt": pred_label})
                print(f"✅ Enviado a /generate: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"❌ Error al conectar con /generate: {e}")

            # Mostrar texto sobre el frame
            text = f"{pred_label} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/main')
def main():
    return render_template('main.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera')
def start_camera():
    global camera_active
    if init_camera():
        camera_active = True
        return jsonify({"status": "success", "message": "Cámara iniciada"})
    return jsonify({"status": "error", "message": "No se pudo iniciar la cámara"})

@app.route('/stop_camera')
def stop_camera():
    global camera_active, cap
    camera_active = False
    if cap is not None:
        cap.release()
        cap = None
    return jsonify({"status": "success", "message": "Cámara detenida"})

@app.route('/gesture_data')
def gesture_data():
    return jsonify({
        "gesture": current_gesture,
        "confidence": current_confidence
    })

if __name__ == '__main__':
    import os
    if os.name == 'nt':
        import multiprocessing
        multiprocessing.freeze_support()

    print("🚀 Iniciando servidor Flask...")
    print("📱 Abre http://localhost:8002 en tu navegador")
    app.run(debug=True, host='0.0.0.0', port=8002)
