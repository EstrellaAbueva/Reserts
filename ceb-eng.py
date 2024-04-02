import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torchaudio
import speech_recognition as sr
import pandas as pd
import torch.nn as nn

# Define the path to the directory containing audio files for transcription
transcription_input_folder = "./audio"
transcription_output_folder = "./transcriptions"

# Define the path to the directory containing audio files for training the BERT model
training_folder = "./audio"

# Define the path to the Excel file containing translations
translations_excel_file = "./sample.xlsx"

# Define the number of output classes based on your classification task
num_classes = 5

# Define the learning rate for the optimizer
learning_rate = 1e-4

# Define the number of epochs for training
num_epochs = 5

# Initialize the recognizer for audio transcription
recognizer = sr.Recognizer()

# Create the output folder for transcriptions if it doesn't exist
if not os.path.exists(transcription_output_folder):
    os.makedirs(transcription_output_folder)

# Load translations from the Excel file
translations_df = pd.read_excel(translations_excel_file)  # Assuming translations are stored in the Excel file
translations_map = dict(zip(translations_df['Phrase'], translations_df['English']))

# Iterate over the files in the transcription input folder
for filename in os.listdir(transcription_input_folder):
    if filename.endswith(".wav"):  # Adjust the file extension if necessary
        input_file_path = os.path.join(transcription_input_folder, filename)
        output_file_path = os.path.join(transcription_output_folder, f"{os.path.splitext(filename)[0]}.txt")
        with sr.AudioFile(input_file_path) as source:
            audio_data = recognizer.record(source)  # Record the entire audio file
            try:
                text = recognizer.recognize_google(audio_data)
                # Skip saving the transcribed text to a file
                print(f"Recognized text from {filename}")
                english_translation = translations_map.get(text, "Translation not available")
                print(f"English Translation: {english_translation}")
            except sr.UnknownValueError:
                print(f"Google Speech Recognition could not understand the audio in {filename}")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service for {filename}; {e}")

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_classes)

# Define optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Audio preprocessing and conversion to text function for BERT model training
def preprocess_audio(file_path):
    audio_input, _ = torchaudio.load(file_path)
    return audio_input

def audio_to_text(audio_input):
    recognizer = sr.Recognizer()
    audio_data = sr.AudioData(audio_input.numpy().tobytes(), sample_width=2, sample_rate=16000)
    try:
        audio_text = recognizer.recognize_google(audio_data)  # Specify the language as Cebuano
    except sr.UnknownValueError:
        audio_text = ""
    return audio_text

# Define your dataset class for BERT model training
class MyDataset(Dataset):
    def __init__(self, folder_path, labels, translations):
        self.audio_files = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.wav')]
        self.labels = labels[:len(self.audio_files)]  # Ensure labels match the number of audio files
        self.translations = translations
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file_path = self.audio_files[idx]
        with open(file_path, "rb") as file:  # Open the file in binary mode
            audio_text = file.read().decode("latin-1")  # Use Latin-1 encoding
        inputs = self.tokenizer(audio_text, padding=True, truncation=True, return_tensors="pt")
        # Adjust the input tensor shape to match BERT's expected format
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}  
        label = self.labels[idx]  # Assuming labels are provided as a list
        translation = self.translations.get(audio_text, "Translation not available")
        return inputs, label, translation
    
# Print the number of classes
print("Number of classes:", num_classes)

# Load labels from Excel file for BERT model training
excel_file = "./sample.xlsx"
df = pd.read_excel(excel_file)
labels = df['Phrase'].tolist()

# Preprocess labels to ensure they only contain numeric values
# For example, if labels are strings, you might need to encode them using label encoding
label_map = {label: idx for idx, label in enumerate(set(labels))}
labels = [label_map[label] for label in labels]
print("Labels after preprocessing:", labels)

# Create an instance of your dataset for BERT model training
dataset = MyDataset(training_folder, labels, translations_map)

# Define the batch size for DataLoader
batch_size = 4

# Create a DataLoader for batching and shuffling the dataset for BERT model training
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Training loop for the BERT model
for epoch in range(num_epochs):
    model.train()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    for batch in dataloader:
        inputs, batch_labels, translations = batch
        inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the appropriate device
        batch_labels = batch_labels.to(device)  # Move labels to the appropriate device
        outputs = model(**inputs)
        logits = outputs.logits
        loss = criterion(logits, batch_labels)

        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Track accuracy
        predictions = torch.argmax(logits, dim=1)
        total_correct += torch.sum(predictions == batch_labels).item()
        total_samples += batch_labels.size(0)

        # Track total loss
        total_loss += loss.item() * batch_labels.size(0)

    # Calculate average loss and accuracy
    epoch_loss = total_loss / total_samples if total_samples > 0 else 0.0
    epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    # Evaluate the model on the validation set (not shown in this example)
    scheduler.step()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}")

# Save the fine-tuned BERT model
model.save_pretrained("fine_tuned_bert_modelCeb-Eng")
print("Model Saved.")
