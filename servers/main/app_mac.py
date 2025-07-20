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

# Intentar cargar el modelo si existe
if TF_AVAILABLE:
    try:
        # Cargar labels primero
        if os.path.exists(LABELS_PATH):
            label_classes = np.load(LABELS_PATH, allow_pickle=True)
            print(f"✅ Labels cargados: {len(label_classes)} gestos")
        else:
            label_classes = ["Rock", "Paper", "Scissors", "Peace", "Fist"]
            
        # Intentar cargar el modelo original primero
        model_loaded = False
        
        if os.path.exists(MODEL_PATH):
            try:
                # Intentar con custom_object_scope para DTypePolicy
                with tf.keras.utils.custom_object_scope({'DTypePolicy': None}):
                    model = load_model(MODEL_PATH, compile=False)
                    print("✅ Modelo original cargado exitosamente")
                    model_loaded = True
            except Exception as e1:
                print(f"❌ Error cargando modelo original: {e1}")
                
                # Intentar el modelo demo
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
        print("⚠️ Modo demo activado")
        model = None
        # Intentar cargar las etiquetas reales aunque el modelo no funcione
        try:
            label_classes = np.load(LABELS_PATH, allow_pickle=True)
            print(f"✅ Labels reales cargados: {len(label_classes)} gestos")
        except:
            label_classes = ["Rock", "Paper", "Scissors", "Peace", "Fist"]
else:
    print("⚠️  TensorFlow no disponible - Modo demo activado")
    model = None
    # Intentar cargar las etiquetas reales aunque TF no esté disponible
    try:
        label_classes = np.load(LABELS_PATH, allow_pickle=True)
        print(f"✅ Labels reales cargados: {len(label_classes)} gestos")
    except:
        label_classes = ["Rock", "Paper", "Scissors", "Peace", "Fist"]

# === Configuración ===
SEQUENCE_LENGTH = 50
NUM_FEATURES = 126  # 21 puntos * 3 coords * 2 manos
sequence = deque(maxlen=SEQUENCE_LENGTH)

# === Inicializar MediaPipe ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.7, min_tracking_confidence=0.5)

# === Variables globales ===
cap = None
camera_active = False
current_gesture = "Sin detectar"
current_confidence = 0.0
hand_positions = []  # Lista de posiciones de centros de manos: [{"x": 0.5, "y": 0.5, "hand": "left/right"}]

def init_camera():
    global cap
    if cap is None:
        # Mac compatibility - usar cv2.VideoCapture sin CAP_DSHOW
        cap = cv2.VideoCapture(0)
    return cap is not None and cap.isOpened()

def generate_frames():
    global camera_active, current_gesture, current_confidence, hand_positions
    
    while camera_active:
        if cap is None or not cap.isOpened():
            break
            
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocesamiento
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Extraer landmarks si hay manos detectadas
        hand_landmarks = []
        current_hand_positions = []
        
        if results.multi_hand_landmarks:
            # Procesar cada mano detectada
            for idx, hand in enumerate(results.multi_hand_landmarks):
                # Calcular centro de la mano (promedio de todos los landmarks)
                center_x = sum([lm.x for lm in hand.landmark]) / len(hand.landmark)
                center_y = sum([lm.y for lm in hand.landmark]) / len(hand.landmark)
                
                # Determinar si es mano izquierda o derecha
                hand_label = "left" if results.multi_handedness[idx].classification[0].label == "Left" else "right"
                
                current_hand_positions.append({
                    "x": center_x,
                    "y": center_y,
                    "hand": hand_label
                })
                
                # Extraer landmarks para el modelo
                for lm in hand.landmark:
                    hand_landmarks.extend([lm.x, lm.y, lm.z])

            # Actualizar posiciones globales
            hand_positions = current_hand_positions

            # Rellenar si solo hay una mano
            if len(results.multi_hand_landmarks) == 1:
                hand_landmarks.extend([0] * 63)  # 21*3=63

            if len(hand_landmarks) == NUM_FEATURES:
                sequence.append(hand_landmarks)

                # Dibujar en pantalla
                for hand in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # Cuando el buffer esté lleno
        if len(sequence) == SEQUENCE_LENGTH:
            if model is not None:
                # Usar el modelo real
                input_data = np.expand_dims(sequence, axis=0)  # (1, 50, 126)
                pred_probs = model.predict(input_data, verbose=0)
                pred_index = np.argmax(pred_probs)
                pred_label = label_classes[pred_index]
                confidence = pred_probs[0][pred_index]
            else:
                # Modo demo - simular detección
                import random
                pred_label = random.choice(label_classes)
                confidence = random.uniform(0.7, 0.95)

            # Actualizar variables globales
            current_gesture = pred_label
            current_confidence = confidence
            
            # Debug: Imprimir para verificar actualización
            print(f"🎯 Gesto detectado: {pred_label} ({confidence:.2f})")

            # Enviar al servidor (primer endpoint)
            try:
                response1 = requests.post("http://localhost:5022/generate", json={"prompt": pred_label}, timeout=2)
                print(f"✅ Enviado a 5022/generate: {response1.status_code} - {response1.text}")
            except Exception as e:
                print(f"❌ Error al conectar con 5022/generate: {e}")

            # Enviar al servidor (segundo endpoint)
            try:
                response2 = requests.post("http://localhost:5009/generate", json={"prompt": pred_label}, timeout=2)
                print(f"✅ Enviado a 5009/generate: {response2.status_code} - {response2.text}")
            except Exception as e:
                print(f"❌ Error al conectar con 5009/generate: {e}")

            # Mostrar texto en la imagen
            text = f"{pred_label} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 0), 3)

        # Codificar frame como JPEG
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
    data = {
        "gesture": current_gesture,
        "confidence": current_confidence
    }
    print(f"📡 Enviando gesture_data: {data}")
    return jsonify(data)

@app.route('/hand_positions')
def get_hand_positions():
    return jsonify({
        "hands": hand_positions
    })

# Crear route dinámico para CSS (necesario para bg.svg)
@app.route('/css/index.css')
def dynamic_css():
    from flask import url_for
    css_content = f"""
    * {{
        margin: 0;
        padding: 0;
    }}

    @font-face {{
        font-family: 'SF Pro Display';
        src: url('{url_for('static', filename='SF-Pro-Display-Regular.otf')}') format('opentype');
        font-weight: 400;
    }}

    @font-face {{
        font-family: 'SF Pro Display';
        src: url('{url_for('static', filename='SF-Pro-Display-Bold.otf')}') format('opentype');
        font-weight: 700;
    }}

    @font-face {{
        font-family: 'SF Pro Display';
        src: url('{url_for('static', filename='SF-Pro-Display-Semibold.otf')}') format('opentype');
        font-weight: 600;
    }}

    body {{
        width: 100%;
        height: 100dvh;
        background: url('{url_for('static', filename='bg.svg')}') no-repeat center center;
        background-size: cover;
        overflow: hidden;
    }}

    .container {{
        position: relative;
        width: 100%;
        height: 100%;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
    }}

    .shadow {{
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.4), transparent 50%, rgba(0, 0, 0, 0.4));
    }}

    .content {{
        position: relative;
        z-index: 2;
        width: 90%;
        height: 90%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: space-between;
        padding: 40px;
    }}

    .text-box {{
        text-align: center;
        color: white;
        font-family: 'SF Pro Display', sans-serif;
    }}

    #text-top {{
        font-size: 4rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 20px;
    }}

    #text-top span:first-child {{
        font-weight: 400;
        font-size: 3rem;
    }}

    #text-top span:last-child {{
        display: block;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient 3s ease infinite;
    }}

    @keyframes gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    #text-bottom {{
        font-size: 1.5rem;
        font-weight: 400;
        margin-top: 20px;
    }}

    #text-bottom button {{
        margin-top: 30px;
        padding: 15px 40px;
        font-size: 1.2rem;
        font-weight: 600;
        font-family: 'SF Pro Display', sans-serif;
        background: rgba(255, 255, 255, 0.1);
        color: white;
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 50px;
        cursor: pointer;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }}

    #text-bottom button:hover {{
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.5);
        transform: translateY(-2px);
    }}

    #airpods {{
        position: absolute;
        right: 10%;
        top: 50%;
        transform: translateY(-50%);
        width: 300px;
        height: auto;
        opacity: 0.8;
    }}
    """
    return css_content, 200, {'Content-Type': 'text/css'}

if __name__ == '__main__':
    print("🚀 Iniciando servidor Flask para Mac...")
    print("📱 Abre http://localhost:8002 en tu navegador")
    print("🔗 Enviará gestos a:")
    print("   - http://localhost:5022/generate")
    print("   - http://localhost:5009/generate")
    app.run(debug=True, host='0.0.0.0', port=8002)