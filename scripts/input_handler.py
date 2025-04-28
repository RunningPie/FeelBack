# input_handler.py
import os
import sounddevice as sd
import pandas as pd
import speech_recognition as sr
from datetime import datetime
from scipy.io.wavfile import write

recognizer = sr.Recognizer()

AUDIO_DIR = "data/for_transcription/"
MIC_DIR = "data/mic_recordings/"
CSV_DIR = "data/processed/"

# Record audio from microphone and save it
def record_from_mic(duration=5, sample_rate=None):
    """
    Record audio from the microphone and save it as a WAV file.
    Arguments:
        duration: Duration of recording in seconds (default=5)
        sample_rate: If None, will use device's default sample rate
    Returns:
        The path of the saved WAV file
    """
    print("Preparing to record from mic...")

    # Auto-detect default device sample rate if not provided
    if sample_rate is None:
        device_info = sd.query_devices(sd.default.device[0], 'input')
        sample_rate = int(device_info['default_samplerate'])
        print(f"Using detected sample rate: {sample_rate} Hz")
    else:
        print(f"Using provided sample rate: {sample_rate} Hz")

    # Record audio
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished!")

    # Save audio
    output_dir = "data/mic_recordings/"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"mic_recording-{datetime.now().strftime('%b_%d-%Hh_%Mm')}.wav"
    save_path = os.path.join(output_dir, filename)
    write(save_path, sample_rate, audio)

    print(f"Audio saved at {save_path}")
    return save_path

# Load all WAV files from a directory
def load_wav_files(directory=AUDIO_DIR):
    files = [file for file in os.listdir(directory) if file.endswith(".wav")]
    paths = [os.path.join(directory, file) for file in files]
    return paths

# Find the latest CSV file in the processed directory
def find_latest_csv(directory=CSV_DIR):
    csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
    if not csv_files:
        return None

    latest_csv = max(csv_files, key=lambda x: os.path.getctime(os.path.join(directory, x)))
    print(f"📄 Found latest CSV: {latest_csv}")
    return os.path.join(directory, latest_csv)

# Load filenames from the latest CSV
def load_files_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    filenames = df['filename'].tolist()
    return filenames
