import asyncio
import wave
import time
from pathlib import Path

from google import genai
from google.genai import types
import aioconsole  # pip install aioconsole
import sounddevice as sd  # pip install sounddevice
import numpy as np

# Output directory
OUTPUT_DIR = Path("generated_audio")
OUTPUT_DIR.mkdir(exist_ok=True)

# Set your API key here
client = genai.Client(
    api_key="AIzaSyAcVa7ksnOPd28Jj-4r8P16ckh_PqYVGBw",
    http_options={'api_version': 'v1alpha'}
)

# Audio parameters
SAMPLE_RATE = 44100
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit audio

# Buffer to store audio chunks for saving
audio_buffer = bytearray()

async def receive_audio(session):
    """Process and play incoming audio while saving it."""
    global audio_buffer
    print("🎧 Ready to receive audio...")
    try:
        async for message in session.receive():
            chunk = message.server_content.audio_chunks[0].data
            audio_buffer.extend(chunk)

            # Convert bytes to numpy int16 array for playback
            audio_np = np.frombuffer(chunk, dtype=np.int16)
            try:
                if not hasattr(receive_audio, "stream"):
                    receive_audio.stream = sd.OutputStream(
                        samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype='int16',
                        blocksize=0 
                    )
                    receive_audio.stream.start()

                receive_audio.stream.write(audio_np)
            except Exception as e:
                print(f"Audio playback error: {e}")

            await asyncio.sleep(0)
    except Exception as e:
        print(f"⚠️ receive_audio stopped: {e}")
        raise  # Let the caller know to reconnect

def save_audio_file(prompt: str):
    """Save buffered audio to a .wav file."""
    global audio_buffer
    if not audio_buffer:
        print("⚠️ No audio data to save.")
        return
    filename = OUTPUT_DIR / f"{int(time.time())}_{prompt.replace(' ', '_')}.wav"
    with wave.open(str(filename), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_buffer)
    print(f"💾 Audio saved to: {filename}")
    audio_buffer = bytearray()  # Reset buffer

async def prompt_listener(session):
    """Listen for user prompts to change music style in real-time."""
    current_prompt = "solo violin playing classical melody"
    while True:
        user_input = await aioconsole.ainput("\nEnter a new prompt (or 'q' to quit): ")
        if user_input.strip().lower() == 'q':
            save_audio_file(current_prompt)
            print("🛑 Exiting.")
            break
        if user_input.strip():
            save_audio_file(current_prompt)
            current_prompt = user_input.strip()
            print(f"🎵 Changing music style to: {current_prompt}")
            await session.set_weighted_prompts(
                prompts=[types.WeightedPrompt(text=f"classical music, {current_prompt}", weight=1.0)]
            )
            await session.play()

async def run_session(initial_prompt="solo violin playing classical melody"):
    """Single session block that handles playback and prompt input."""
    print(f"🎼 Starting session with initial prompt: {initial_prompt}")
    async with (
        client.aio.live.music.connect(model='models/lyria-realtime-exp') as session,
        asyncio.TaskGroup() as tg,
    ):

        await session.set_weighted_prompts(
            prompts=[types.WeightedPrompt(text=initial_prompt, weight=1.0)]
        )
        await session.set_music_generation_config(
            config=types.LiveMusicGenerationConfig(bpm=120, temperature=1.0)
        )
        await session.play()

        tg.create_task(receive_audio(session))
        tg.create_task(prompt_listener(session))

async def main():
    initial_prompt = "violin alegre"
    while True:
        try:
            await run_session(initial_prompt)
            break  # Exit loop if run_session completes normally (e.g., user quits)
        except Exception as e:
            print(f"💥 Session crashed, restarting with prompt '{initial_prompt}'... Error: {type(e).__name__} - {e}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main())
