import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Path where the transcriptions are stored
TRANSCRIPTION_DIR = "data/processed/audio_transcription/"

# Find all CSV files in the directory
csv_files = [f for f in os.listdir(TRANSCRIPTION_DIR) if f.startswith("audio_transcription")]

# Extract datetime information from filenames
def extract_datetime(file_name):
    match = re.search(r'(\w+)_(\d+)h_(\d+)m', file_name)
    if match:
        hour = int(match.group(2))
        minute = int(match.group(3))
        return hour * 60 + minute  # Convert to minutes for easier comparison
    else:
        return -1  # Files that don't match will be considered oldest

def clusterize_transcriptions():
    # Sort files by extracted time
    csv_files_sorted = sorted(csv_files, key=lambda x: extract_datetime(x), reverse=True)

    # Pick the latest file
    latest_file = csv_files_sorted[-1] if csv_files_sorted else None

    if latest_file:
        latest_file_path = os.path.join(TRANSCRIPTION_DIR, latest_file)
        print(f"Latest transcription file found: {latest_file_path}")
    else:
        raise FileNotFoundError("No transcription CSV files found in the processed directory.")

    # Read the CSV file
    df = pd.read_csv(latest_file_path)
    print("Total rows:", len(df))
    print("Sample:\n", df.head())

    unique_df = df.drop_duplicates(subset=["transcription"])
    texts = unique_df["transcription"].values
    print(f"Unique transcriptions: {len(texts)}")

    # Vectorize Text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    print("Vectorized shape:", X.shape)

    inertia = []

    # Test K from 1 to unique//2 because in unique//2 each cluster would
    # approximately have 2 samples and by then it would probably
    # already be useless to even cluster it
    k_values = range(1, len(texts)//2)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    # Handle case where inertia is too small for a meaningful elbow
    if len(inertia) < 2:
        print("Not enough unique transcriptions for meaningful clustering. Skipping elbow method.")
        k_optimal = 1  # Default to 1 cluster
    else:
        # Plot the Elbow
        plt.figure(figsize=(8,6))
        plt.plot(k_values, inertia, marker='o')
        plt.title('Elbow Method For Optimal k')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia (Sum of Squared Distances)')
        plt.grid(True)
        plt.show()

        # Compute the difference and find the optimal k
        diffs = np.diff(inertia)
        k_optimal = np.argmin(diffs) + 2  # +2 because diff array is len-1

    print(f"Suggested optimal number of clusters: {k_optimal}")

    # Apply KMeans with Optimal K
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    kmeans.fit(X)
    unique_df["cluster"] = kmeans.labels_

    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(X.toarray())

    # Visualize Clusters
    plt.figure(figsize=(8,6))
    for i in range(k_optimal):
        points = reduced[unique_df['cluster'] == i]
        plt.scatter(points[:, 0], points[:, 1], label=f"Cluster {i}")
    plt.title(f"K-Means Clustering of Audio Transcriptions (k={k_optimal})")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nExample sentences by cluster:")
    for i in range(k_optimal):
        print(f"\n--- Cluster {i} ---")
        samples = unique_df[unique_df["cluster"] == i]["transcription"]
        print(samples.sample(min(3, len(samples)), random_state=42).tolist())
