import numpy as np
import torch
from models.generator import Generator
from util import Audio
import os


test_folder = './data/test'  
output_folder = './outputs'  
os.makedirs(output_folder, exist_ok=True)


def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    gen = Generator()
    gen.load_state_dict(checkpoint['gen'])
    gen.eval()
    return gen


def generate_audio_from_test(model, test_folder, output_folder):
    audio = Audio()
    for file in os.listdir(test_folder):
        if file.endswith('.npy'):
            genre = file.split('.')[0]  
            print(f"Processing genre: {genre}")

 
            mel_data = np.load(os.path.join(test_folder, file))

            for i, mel in enumerate(mel_data):
                mel_tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0)  
                with torch.no_grad():
                    generated = model(mel_tensor)  
                output_audio = audio.mel_to_audio(generated.squeeze().numpy())
                output_path = os.path.join(output_folder, f"{genre}_generated_{i}.wav")
                audio.save_audio(output_audio, output_path)
                print(f"Saved: {output_path}")

# Main
checkpoint_path = './checkpoints/checkpoint_100.pt' 
generator_model = load_model(checkpoint_path)
generate_audio_from_test(generator_model, test_folder, output_folder)
