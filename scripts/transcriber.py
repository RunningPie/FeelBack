# transcriber.py
import os
import concurrent.futures
import pandas as pd
import speech_recognition as sr
from datetime import datetime

recognizer = sr.Recognizer()

# Transcribe a single audio file
def transcribe_file(file_path):
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            print(f"✅ Transcribed: {os.path.basename(file_path)}")
    except sr.UnknownValueError:
        text = "[Unintelligible]"
        print(f"🤔 Could not understand: {os.path.basename(file_path)}")
    except sr.RequestError as e:
        text = "[API Error]"
        print(f"❌ API error for {os.path.basename(file_path)}: {e}")

    return {"filename": os.path.basename(file_path), "transcription": text}

# Batch transcribe with multithreading
def batch_transcribe(file_paths, max_workers=8, save=True):
    start_time = datetime.now()

    if not file_paths:
        print("❗No files to transcribe.")
        return []

    max_workers = min(max_workers, len(file_paths))
    print(f"🚀 Starting transcription with {max_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(transcribe_file, file_paths))

    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()

    if execution_time > 60:
        print(f"⏱️ Total time: {execution_time//60}mins and {execution_time%60:.2f}secs")
    else:
        print(f"⏱️ Total time: {execution_time:.2f}s")

    if save:
        process_path = "data/processed/audio_transcription"
        os.makedirs(process_path, exist_ok=True)
        filename = f"audio_transcription-{datetime.now().strftime('%b_%d-%Hh_%Mm')}.csv"
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(process_path, filename), index=False)
        print(f"📦 Saved transcription to {filename}")

    return filename
