import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import torchaudio
from torchaudio.transforms import Resample

# Load data
data = pd.read_csv("Spotify-2000.csv")
data = data.drop("Index", axis=1)

# Display correlation matrix
print(data.corr())

# Select relevant features
features = ["Beats Per Minute (BPM)", "Loudness (dB)", "Liveness", "Valence", "Acousticness", "Speechiness"]
data2 = data[features]

# Standardize features
scaler = StandardScaler()
data2_scaled = scaler.fit_transform(data2)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42)
data["Music Segments"] = kmeans.fit_predict(data2_scaled)

# Map cluster labels to human-readable names
data["Music Segments"] = data["Music Segments"].map(lambda x: f"Cluster {x + 1}")

# Plot 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster_label in data["Music Segments"].unique():
    cluster_data = data[data["Music Segments"] == cluster_label]
    ax.scatter(cluster_data['Beats Per Minute (BPM)'], cluster_data['Energy'], cluster_data['Danceability'],
               label=cluster_label, s=30)

ax.set_xlabel('Beats Per Minute (BPM)')
ax.set_ylabel('Energy')
ax.set_zlabel('Danceability')
ax.set_title('Music Segments Clustering')

plt.legend()
plt.show()

# Example of using torchaudio for audio processing
filename = 'path_to_audio_file.mp3'
waveform, sample_rate = torchaudio.load(filename)
resampler = Resample(orig_freq=sample_rate, new_freq=8000)
resampled_waveform = resampler(waveform)

# Example of using a simple autoencoder for clustering
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = torch.relu(self.encoder(x))
        x = torch.relu(self.decoder(x))
        return x

# Convert data to PyTorch tensors
data_tensor = torch.tensor(data2_scaled, dtype=torch.float32)

# Instantiate and train the autoencoder
input_size = len(features)
hidden_size = 3  # You can adjust this based on your preference
autoencoder = Autoencoder(input_size, hidden_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    output = autoencoder(data_tensor)
    loss = criterion(output, data_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Obtain encoded features from the autoencoder
encoded_features = autoencoder.encoder(data_tensor).detach().numpy()

# Apply KMeans clustering to the encoded features
kmeans_autoencoder = KMeans(n_clusters=10, random_state=42)
data["Music Segments Autoencoder"] = kmeans_autoencoder.fit_predict(encoded_features)

# Map cluster labels to human-readable names
data["Music Segments Autoencoder"] = data["Music Segments Autoencoder"].map(lambda x: f"Cluster {x + 1}")

# Print the resulting dataframe
print(data.head())
