import torch
import os
import torchaudio
from transformers import BertTokenizer, BertForSequenceClassification
import speech_recognition as sr

# Load the fine-tuned BERT model
model_path = "./fine_tuned_bert_modelWithAccuracy"
model = BertForSequenceClassification.from_pretrained(model_path)
# Load the fine-tuned BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)

# Define the path to the folder containing audio files
folder_path = "./audio"

# Function to transcribe audio file using SpeechRecognition library
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()

    # Load audio file
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)  # Read the entire audio file

    # Transcribe audio using Google Speech Recognition
    try:
        transcription = recognizer.recognize_google(audio, language="ceb")  # Specify the language as Cebuano
        return transcription
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

    return None

# Iterate over all audio files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):  # Adjust the file extension if necessary
        audio_file = os.path.join(folder_path, filename)
        
        # Transcribe the audio file
        transcription = transcribe_audio(audio_file)
        print("Transcription result:", transcription)  # Debugging statement
        if transcription:
            # Tokenize the transcribed text using the fine-tuned tokenizer
            inputs = tokenizer(transcription, padding=True, truncation=True, return_tensors="pt")
            
            # Perform inference
            with torch.no_grad():
                outputs = model(**inputs)

            # Interpret the results
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            predicted_probability = torch.softmax(logits, dim=1).tolist()[0]

            print(f"Audio File: {filename}")
            print("Transcription:", transcription)
            print("Predicted Class:", predicted_class)
            print("Predicted Probabilities:", predicted_probability)
        else:
            print(f"Transcription failed for audio file: {filename}")