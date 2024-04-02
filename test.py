import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification
import torchaudio
import speech_recognition as sr
from mtranslate import translate as mtranslate

# Load the fine-tuned BERT tokenizer for Cebuano to English translation
tokenizer = BertTokenizer.from_pretrained("./fine_tuned_bert_modelCeb-Eng")

# Load the fine-tuned BERT model for Cebuano to English translation
model = BertForSequenceClassification.from_pretrained("./fine_tuned_bert_modelCeb-Eng")

# Function to transcribe audio file
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()

    # Load audio file
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)  # Read the entire audio file

    # Transcribe audio using Google Speech Recognition
    try:
        transcription = recognizer.recognize_google(audio)
        return transcription
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    return None

# Function to translate text
def translate_text(text, target_language='en'):
    try:
        translated_text = mtranslate(text, target_language)
        return translated_text
    except Exception as e:
        print(f"Translation failed: {e}")
        return None

# Transcribe and translate audio folder
def transcribe_and_translate_folder(folder_path):
    audio_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]
    translations = {}
    for audio_file in audio_files:
        transcription = transcribe_audio(audio_file)
        if transcription:
            translation = translate_text(transcription)
            if translation:
                translations[os.path.basename(audio_file)] = translation
            else:
                translations[os.path.basename(audio_file)] = "Translation failed."
        else:
            translations[os.path.basename(audio_file)] = "Transcription failed."
    return translations

# Example usage
folder_path = "./audio"
translations = transcribe_and_translate_folder(folder_path)
for audio_file, translation in translations.items():
    print(f"Audio File: {audio_file}, Translation: {translation}")
