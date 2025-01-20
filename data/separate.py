import os
import subprocess
import shutil
from pydub import AudioSegment


def source_separate(audio_dir, output_dir):
    """
    Separates audio files into components (bass, drums, vocals, other) using demucs.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Command to run Demucs for separation
    command = [r"path_to_python.exe", "-m", "demucs.separate", "-n", "htdemucs"]

    # Process each genre folder (e.g., country, reggae)
    for genre in os.listdir(audio_dir):
        genre_dir = os.path.join(audio_dir, genre)

        # Skip non-directory files in the audio_dir
        if not os.path.isdir(genre_dir):
            continue

        output_genre_dir = os.path.join(output_dir, genre)
        os.makedirs(output_genre_dir, exist_ok=True)

        # Process each .wav file in the genre directory
        for fname in os.listdir(genre_dir):
            if not fname.endswith('.wav'):
                continue  # Skip non-wav files

            # Full path to the input file
            input_file = os.path.join(genre_dir, fname)

            # Run Demucs for separation
            cmd = command + [input_file]
            subprocess.run(cmd)

            # Organize separated files by component (bass, drums, etc.)
            name = os.path.splitext(fname)[0]  # Get file name without extension
            separated_path = os.path.join("separated", "demucs_quantized", name)

            if os.path.exists(separated_path):
                # Move separated components to the output directory
                for component in ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]:
                    component_path = os.path.join(separated_path, component)
                    if os.path.exists(component_path):
                        component_output_dir = os.path.join(output_genre_dir, component.split('.')[0])  # bass, drums, etc.
                        os.makedirs(component_output_dir, exist_ok=True)
                        shutil.move(component_path, os.path.join(component_output_dir, f"{name}.wav"))

                # Clean up temporary separated files
                shutil.rmtree(separated_path)


if __name__ == "__main__":
    import argparse

    # Parse input and output directories
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, required=True, help="Directory containing original audio files by genre")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to store separated components")
    args = parser.parse_args()

    # Perform source separation
    source_separate(args.audio_dir, args.output_dir)
