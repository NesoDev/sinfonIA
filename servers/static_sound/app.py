from flask import Flask, request, jsonify
import os
import threading
from playsound import playsound  # Make sure this is installed: pip install playsound==1.2.2
from pydub import AudioSegment
from pydub.playback import play

app = Flask(__name__)

# Path to your sound file
SOUND_PATH = os.path.join("sounds", "violin_plus_chelo.wav")



import simpleaudio as sa

def play_sound():
    try:
        sound = AudioSegment.from_wav(SOUND_PATH)
        play_obj = sa.play_buffer(
            sound.raw_data,
            num_channels=sound.channels,
            bytes_per_sample=sound.sample_width,
            sample_rate=sound.frame_rate
        )
        play_obj.wait_done()
    except Exception as e:
        print(f"⚠️ Error playing sound: {e}")



@app.route('/generate', methods=['POST'])
def generate_sound():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    print(f"🔊 Received prompt: {prompt}")

    # Play sound in a separate thread
    threading.Thread(target=play_sound).start()

    return jsonify({"status": "Sound triggered", "prompt": prompt}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5009)
