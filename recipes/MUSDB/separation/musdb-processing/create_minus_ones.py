import os
import subprocess

# Define source and destination paths
source_path = "/data3/jiaqi.yip/musdb18_stems/8k"

# Define stems to convert
stems = ['other.wav', 'linear_mixture.wav', 'mixture.wav', 'drums.wav', 'accompaniment.wav', 'bass.wav']
vocal_stem = 'vocals.wav'  # Identify the vocal stem

def create_minus_one(song_path):
  # Create the output path for minus_one.wav
  minus_one_output = os.path.join(song_path, 'minus_one.wav')

  minus_one_inputs = []  # List to store input files for minus_one.wav
  for stem in stems:
    source_file = os.path.join(song_path, stem)
    
    # Check if file exists and has a `.wav` extension
    if os.path.isfile(source_file) and stem.endswith('.wav') and stem != vocal_stem:
      minus_one_inputs.append(source_file)  # Add non-vocal stems

  # Create minus_one.wav (if there are non-vocal stems)
  if minus_one_inputs:
    command = ['ffmpeg', '-i', 'concat:' + '|'.join(minus_one_inputs), '-acodec', 'copy', minus_one_output]
    subprocess.run(command, check=True)
    print(f"Created minus_one.wav: {minus_one_output}")

# Loop through each song folder
for dataset in os.listdir(source_path):
  # Skip hidden folders
  if dataset.startswith('.'):
    continue
  dataset_path = os.path.join(source_path, dataset)
  for song in os.listdir(dataset_path):
    song_path = os.path.join(dataset_path, song)
    create_minus_one(song_path)
