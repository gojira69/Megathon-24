import os
import librosa
import pandas as pd
from transformers import pipeline
from pydub import AudioSegment
import sys

# Set up ASR pipeline for transcription
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-medium", device=-1)
asr_pipeline.model.config.forced_decoder_ids = asr_pipeline.tokenizer.get_decoder_prompt_ids(language="en", task="transcribe")

# Function to convert MP3 to WAV and transcribe the audio
def transcribe_audio_file(mp3_file_path, output_txt_path):
    # Convert MP3 to WAV
    audio = AudioSegment.from_mp3(mp3_file_path)
    wav_path = os.path.splitext(mp3_file_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    print(f"Converted {mp3_file_path} to {wav_path}")
    
    # Load WAV audio and transcribe
    audio_data, _ = librosa.load(wav_path, sr=16000)
    transcription = asr_pipeline(audio_data)["text"]
    print(f"Transcription: {transcription}")
    
    # Save transcription to text file
    with open(output_txt_path, "w") as f:
        f.write(transcription)
    print(f"Transcription saved to {output_txt_path}")

def main():
    username = sys.argv[1]
    data_folder = "Data"
    mp3_file_path = os.path.join(data_folder, username, "audio.mp3")  # Adjust the file path as needed
    output_txt_path = os.path.join(data_folder, username, "therapists.txt")  # Output file path

    # Check if the audio file exists
    if not os.path.exists(mp3_file_path):
        print(f"Error: The file {mp3_file_path} does not exist.")
        return

    # Transcribe the audio file
    transcribe_audio_file(mp3_file_path, output_txt_path)

if __name__ == "__main__":
    main()
