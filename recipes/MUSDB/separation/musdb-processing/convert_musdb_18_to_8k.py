import os
import subprocess
from tqdm import tqdm

# Define source and destination paths
source_path = "/data3/jiaqi.yip/musdb18_stems/44k"
destination_path = "/data3/jiaqi.yip/musdb18_stems/8k"

# Define stems to convert
stems = ['other.wav', 'linear_mixture.wav', 'mixture.wav', 'drums.wav', 'accompaniment.wav', 'bass.wav', 'vocals.wav']

def copy_and_convert(source, destination):
  # Create the destination directory if it doesn't exist
  os.makedirs(destination, exist_ok=True)

  for dataset in tqdm(os.listdir(source)):
    # Skip hidden folders
    if dataset.startswith('.'):
      continue
    source_dataset_path = os.path.join(source, dataset)
    destination_dataset_path = os.path.join(destination, dataset)
    
    # Create the dataset folder in destination
    os.makedirs(destination_dataset_path, exist_ok=True)

    for song in tqdm(os.listdir(source_dataset_path)):
      source_song_path = os.path.join(source_dataset_path, song)
      destination_song_path = os.path.join(destination_dataset_path, song)
      
      # Create the song folder in destination
      os.makedirs(destination_song_path, exist_ok=True)

      for stem in stems:
        source_file = os.path.join(source_song_path, stem)
        destination_file = os.path.join(destination_song_path, stem)

        # Check if file exists and has a `.wav` extension
        if os.path.isfile(source_file) and stem.endswith('.wav'):
          # Convert and copy the file
          command = ['ffmpeg', '-i', source_file, '-ar', '8000', destination_file]
          subprocess.run(command, check=True)
          print(f"Converted: {source_file} -> {destination_file}")
        else:
          # Copy non-wav files directly
          shutil.copy2(source_file, destination_file)  # Import shutil for copy2

# Call the function
copy_and_convert(source_path, destination_path)
