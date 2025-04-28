
# Audio Transcriber + Emotion Predictor

This project provides an easy-to-use tool for transcribing audio files and predicting emotions from speech. It supports two main features: recording audio from the microphone, transcribing the speech, and predicting the emotion of the speech, or clustering transcription data using K-Means clustering.

## Features

1. **Record and Predict Emotion from Mic**: Record an audio clip via the microphone, transcribe it, and predict the emotion of the speech.
2. **Clusterize WAV Files**: Cluster audio transcriptions into groups based on their content using K-Means clustering. 

## Requirements

To run this project, you will need:

- Python 3.x
- Libraries: 
  - `librosa` for audio processing
  - `numpy` for numerical computations
  - `pandas` for data manipulation
  - `sklearn` for machine learning (K-Means clustering)
  - `matplotlib` for plotting visualizations
  - `soundfile` for reading audio files
  - `speech_recognition` for transcription from audio
  - `pydub` (optional, for audio format conversions)
  - `scipy` for signal processing
  - `time` for managing timing in tasks

You can install the required libraries by running:

```bash
pip install -r requirements.txt
```

## Project Structure

- `notebooks/`: Folder where all notebooks used for model training are stored.
- `models/`: Folder where all saved models of the notebooks are stored.
- `data/`: Folder where all audio files are stored.
  - `mic_recordings/`: Folder for storing audio files recorded via the mic.
  - `for_transcription/`: Folder for audio files ready to be transcribed.
  - `model_training/`: Folder for audio files used in the emotion predictor model training.
  - `processed/`: Folder containing processed transcription files.
- `scripts/`: Contains all the Python scripts used in the project.
  - `input_handler.py`: Handles audio file loading and recording.
  - `transcriber.py`: Contains functions to transcribe audio files.
  - `emotion_predictor.py`: Contains emotion prediction logic for audio files.
  - `clusterize.py`: Contains clustering logic for audio transcriptions.

## How to Use

### 1. Running the Program

To run the program, navigate to the root folder of the project and execute `main.py`:

```bash
python main.py
```

### 2. Choosing Between Options

When you run the program, you will be presented with two choices:

1. **Record and Predict Emotion from Mic**: This option allows you to record an audio clip from your microphone, transcribe it, and predict the emotion in the speech. 
2. **Clusterize WAV Files**: This option allows you to perform K-Means clustering on the transcriptions of multiple WAV files. This will group similar transcriptions together.

### 3. Option 1: Record and Predict Emotion

To record and predict emotion from a mic recording:

- Choose option `1` when prompted.
- The program will record a short audio clip, transcribe it, and predict the emotion in the recorded speech.
  
After recording, the audio file will be stored in `data/mic_recordings`. If you'd like to cluster this recording later, you need to move it manually to the `data/for_transcriptions` folder.

### 4. Option 2: Clusterize WAV Files

To clusterize WAV files (including transcriptions):

- Choose option `2` when prompted.
- The program will look for WAV files in the `data/for_transcriptions/` folder.

Once the WAV files are in the correct folder, the program will process the transcriptions and perform clustering using K-Means. The clustering results will be displayed visually.

### 5. Transcription and Clustering Workflow

If you choose to cluster WAV files, the program will:

- First, transcribe all WAV files in `data/for_transcriptions/`.
- Then, it will perform K-Means clustering on the transcriptions.
- The optimal number of clusters is determined using the Elbow Method, and the transcriptions are grouped accordingly.

## Notes

- Ensure that you have enough WAV files in the `data/for_transcriptions/` folder for clustering (at least 2 files). If there’s only one file, clustering may not be necessary.
- For transcription, audio files in WAV format are required. Ensure your files are in this format before running the transcription.
- **IMPORTANT** You can also clusterize your recorded audio files, but currently you have to manually move or copy-paste the recordings from `data/mic_recordings/` to `data/for_transcription/`
- Also, hopefully in the future I can add model options for the input too :D

## Example Usage

```bash
$ python main.py
🎙️ Welcome to Audio Transcriber + Emotion Predictor 🎙️
Choose input method:
1. Record and Predict Emotion from Mic
2. Clusterize WAV files from data/for_transcription/ Folder
Enter your choice (1/2): 1
Recording... Please speak clearly into the microphone.
Transcription complete. Predicting emotion...
Emotion predicted: Happy!

OR

$ python main.py
🎙️ Welcome to Audio Transcriber + Emotion Predictor 🎙️
Choose input method:
1. Record and Predict Emotion from Mic
2. Clusterize WAV files from data/for_transcription/ Folder
Enter your choice (1/2): 2
Clustering transcriptions...
```

## Troubleshooting

- **No WAV files detected**: If the program can’t find any WAV files in the `data/for_transcriptions/` folder, make sure you’ve moved the files there manually or recorded a new file.
- **Clustering doesn't work**: Make sure there are at least 2 unique transcriptions in the `data/processed/` folder. If there’s only 1, clustering won’t be meaningful.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
