from scripts.input_handler import record_from_mic, load_wav_files, find_latest_csv, load_files_from_csv
from scripts.transcriber import batch_transcribe
from scripts.emotion_predictor import batch_predict_emotion
from scripts.clusterize import clusterize_transcriptions  # Assuming you'll implement the clusterize functionality here
import os
import time

def wait_for_file(file_path, timeout=10):
    """Wait until the file is fully saved (not being modified)"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            return True
        time.sleep(1)
    return False

def main():
    print("🎙️ Welcome to Emotion Predictor + Audio Clusterizer 🎙️")
    print("Choose input method:")
    print("1. Record and Predict Emotion from Mic")
    print("2. Clusterize WAV files from data/for_transcription/ Folder")
    
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        # Option 1: Record and Predict Emotion from Mic
        mic_audio_path = record_from_mic()
        if mic_audio_path:
            files = [mic_audio_path]
            batch_transcribe(files)
            batch_predict_emotion(files)
        else:
            print("❗ No audio recorded. Please try again.")

    elif choice == "2":
        # Option 2: Clusterize WAV files (or transcriptions)
        files = load_wav_files()
        if len(files)>2:
            transcription_file = batch_transcribe(files)
            if wait_for_file("data/processed/audio_transcription/"+transcription_file):
                clusterize_transcriptions()
            else:
                print("❗ Timeout while waiting for transcription file.")
        elif len(files) == 1:
            print("There's only 1 WAV File.. I don't think you need to cluster that :D")
        else:
            print("❗ No WAV file detected. Please try again.")
            
    else:
        print("❌ Invalid choice. Please run the program again and select 1 or 2.")
        return

if __name__ == "__main__":
    main()
