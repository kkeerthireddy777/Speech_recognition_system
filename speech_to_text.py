import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(device)
model.eval()

def transcribe(audio_path):
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    print(f"Original shape: {waveform.shape}, Sample rate: {sr}")

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    print(f"Processed shape: {waveform.shape}")

    # Tokenize
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )

    # Predict
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode
    transcription = processor.batch_decode(predicted_ids)[0]
    print("Transcription:", transcription)

# Example usage
transcribe("harvard.wav")