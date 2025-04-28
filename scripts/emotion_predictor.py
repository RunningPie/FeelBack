# scripts/emotion_predictor.py

from scripts.input_handler import find_latest_csv, load_files_from_csv
import joblib
import librosa
import numpy as np
import pandas as pd
import os
import soundfile as sf

# Load your emotion classification model
model = joblib.load("models/emotion_model-Apr_25-17h_03m.pkl")

def extract_features(file_name, mfcc=True, chroma=True, mel=True):
    """Extract features from an audio file for emotion prediction purposes."""
    with sf.SoundFile(file_name) as sound_file:
        # Read the audio data and sample rate
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        
        result = np.array([])  # Initialize an empty array to store features

        # Compute Chroma features
        if chroma:
            stft = np.abs(librosa.stft(X))  # Short-time Fourier transform
            chroma_vals = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_vals))

        # Compute MFCC features
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        # Compute Mel spectrogram features
        if mel:
            mel_vals = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_vals))

        return result

def batch_predict_emotion(file_paths):
    """Predict emotion for a list of audio files."""
    print("\n🎯 Predicting Emotions for the given files...")
    predictions = []
    
    for file in file_paths:
        try:
            features = extract_features(file)
            prediction = model.predict([features])[0]
            print(f"{file} ➡️ {prediction}")
            predictions.append({"filename": os.path.basename(file), "predicted_emotion": prediction})
        except Exception as e:
            print(f"❗Error processing {file}: {e}")
    
    print(f"Prediction results, your voice sounds: {prediction}")
    
    # Save predictions if needed
    if predictions:
        df = pd.DataFrame(predictions)
        output_filename = f"data/processed/emotion_predictions/emotion_prediction-{pd.Timestamp.now().strftime('%b_%d-%Hh_%Mm')}.csv"
        df.to_csv(output_filename, index=False)
        print(f"✅ Emotion predictions saved to {output_filename}")

def predict_emotion_from_latest_csv():
    """Predict emotion directly from the latest CSV (transcriptions)."""
    latest_csv = find_latest_csv()
    if latest_csv:
        print(f"\n📄 Found latest CSV: {latest_csv}")
        filenames = load_files_from_csv(latest_csv)
        file_paths = ["data/raw/" + file for file in filenames]
        batch_predict_emotion(file_paths)
    else:
        print("❗No latest transcription CSV found.")
