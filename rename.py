import os

# Specify the directory containing the audio files
transcription_input_folder = "./1-100"
output_folder = "./train"  # Specify the output folder

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over the files in the directory and rename them
for i, filename in enumerate(os.listdir(transcription_input_folder), start=1):
    if filename.endswith(".wav"):
        old_path = os.path.join(transcription_input_folder, filename)
        new_filename = f"{i}.wav"
        new_path = os.path.join(output_folder, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed {filename} to {new_filename}")
