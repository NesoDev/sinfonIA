import asyncio
import wave
import time
import threading
from pathlib import Path
from flask import Flask, request, jsonify

from google import genai
from google.genai import types
import numpy as np
import sounddevice as sd

# Flask app
app = Flask(__name__)

# Google client
client = genai.Client(
    api_key="AIzaSyAcVa7ksnOPd28Jj-4r8P16ckh_PqYVGBw",
    http_options={'api_version': 'v1alpha'}
)

# Audio settings
SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit audio

# Output directory
OUTPUT_DIR = Path("generated_audio")
OUTPUT_DIR.mkdir(exist_ok=True)

# Shared buffer and lock
audio_buffer = bytearray()
buffer_lock = threading.Lock()

def save_audio_file(prompt: str):
    """Save the buffer to a WAV file."""
    with buffer_lock:
        if not audio_buffer:
            return None
        filename = OUTPUT_DIR / f"{int(time.time())}_{prompt.replace(' ', '_')}.wav"
        with wave.open(str(filename), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_WIDTH)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_buffer)
        audio_buffer.clear()
        return str(filename)

async def receive_audio(session):
    """Handle incoming audio chunks and playback."""
    global audio_buffer
    try:
        async for message in session.receive():
            chunk = message.server_content.audio_chunks[0].data
            with buffer_lock:
                audio_buffer.extend(chunk)
            audio_np = np.frombuffer(chunk, dtype=np.int16)
            try:
                sd.play(audio_np, samplerate=SAMPLE_RATE)
                sd.wait()
            except Exception as e:
                print(f"Playback error: {e}")
    except Exception as e:
        print(f"Audio receive error: {e}")

async def generate_music(prompt: str):
    """Core generation logic for a single prompt."""
    try:
        async with (
            client.aio.live.music.connect(model='models/lyria-realtime-exp') as session,
            asyncio.TaskGroup() as tg,
        ):
            await session.set_weighted_prompts(
                prompts=[types.WeightedPrompt(text=f"classical music, {prompt}", weight=1.0)]
            )
            await session.set_music_generation_config(
                config=types.LiveMusicGenerationConfig(bpm=90, temperature=1.0)
            )
            await session.play()

            tg.create_task(receive_audio(session))

            # Run for 5 seconds only
            await asyncio.sleep(5)
            await session.stop()

    except Exception as e:
        print(f"Generation error: {e}")

def run_async_generation(prompt: str):
    """Launch the async process from Flask in a thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(generate_music(prompt))
    loop.close()

@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "Missing prompt"}), 400

    # Clear previous audio
    global audio_buffer
    with buffer_lock:
        audio_buffer.clear()

    # Run generation in separate thread
    thread = threading.Thread(target=run_async_generation, args=(prompt,))
    thread.start()
    thread.join()

    file_path = save_audio_file(prompt)
    if not file_path:
        return jsonify({"error": "No audio generated"}), 500

    return jsonify({"message": "Audio generated", "file_path": file_path})

if __name__ == "__main__":
    app.run(debug=True, port=5022)

