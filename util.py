import torch
import numpy as np
import random
import os
import soundfile as sf

class Audio():

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # Load pre-trained speaker encoder and vocoder (WaveGlow)
        gru_embedder = torch.hub.load('RF5/simple-speaker-embedding', 'gru_embedder')
        gru_embedder = gru_embedder.to(self.device)
        gru_embedder.eval()
        self.speaker_encoder = gru_embedder
        
        waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
        waveglow = waveglow.remove_weightnorm(waveglow)
        waveglow = waveglow.to(self.device)
        waveglow.eval()
        self.vocoder = waveglow

    def audio_to_mel(self, audio_file):
        """
        Convert an audio file to a Mel-spectrogram
        """
        mel = self.speaker_encoder.melspec_from_file(audio_file)
        mel = mel.transpose(-1, -2)
        return mel.data.cpu().numpy()

    def mel_to_audio(self, mel):
        """
        Convert Mel-spectrogram back to audio
        """
        mel = np.expand_dims(mel, axis=0)
        mel = torch.from_numpy(mel)
        mel = mel.to(self.device)
        with torch.no_grad():
            audio = self.vocoder.infer(mel)
        return audio[0].data.cpu().numpy()

    def mel_sample(self, mel, width=128, k=5):
        """
        Sample k sections from the Mel-spectrogram
        """
        mel_width = mel.shape[1]
        if mel_width < width:
            return None
        pos = random.choices(range(mel_width - width), k=k)
        samples = np.array([mel[:, x: x + width] for x in pos])
        return samples

    def save_audio(self, audio, output_file):
        """
        Save the generated audio to a file.
        """
        sf.write(output_file, audio, 44100) 

    def process_and_save(self, input_audio, output_audio, width=128, k=5):
        """
        Process the input audio, sample from the Mel-spectrogram, and save the result.
        """
        mel = self.audio_to_mel(input_audio)
        sampled_mels = self.mel_sample(mel, width=width, k=k)
        for idx, sampled_mel in enumerate(sampled_mels):
            audio = self.mel_to_audio(sampled_mel)
            output_file = f"{output_audio}_sampled_{idx}.wav"
            self.save_audio(audio, output_file)
            print(f"Saved sampled audio to {output_file}")

# # Example of usage:
# audio_processor = AudioProcessor()

# input_audio_file = 'separated/htdemucs/country.00000/bass.wav'

# output_audio_file = 'processed_audio_output/bass'

# audio_processor.process_and_save(input_audio_file, output_audio_file)
