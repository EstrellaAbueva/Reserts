import os
import pandas as pd
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import MarianTokenizer, MarianMTModel

# Load the audio dataset
audio_dataset = load_dataset("./audio")

# Load the Excel file containing Cebuano phrases and English translations
def load_excel_file(file_path):
    df = pd.read_excel(file_path)
    cebuano_phrases = df['Phrase'].tolist()
    english_translations = df['English'].tolist()
    return cebuano_phrases, english_translations

excel_file_path = "sample.xlsx"
cebuano_phrases, english_translations = load_excel_file(excel_file_path)

# Create translation dataset from the Excel file
translation_dataset = Dataset.from_dict({"translation": english_translations})

# Load the audio processor and model for speech recognition
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-100h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-100h")

# Load the translation tokenizer and model for translation
translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-ceb-en")
translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-ceb-en")

# Preprocess the datasets
def preprocess_audio(example):
    speech_input_values = processor(example["speech"], return_tensors="pt").input_values
    example["input_values"] = speech_input_values
    return example

def preprocess_translation(example):
    translated_text = translation_tokenizer(example["translation"], return_tensors="pt", padding=True, truncation=True).input_ids
    example["labels"] = translated_text
    return example

audio_dataset = audio_dataset.map(preprocess_audio)
translation_dataset = translation_dataset.map(preprocess_translation)

# Combine the datasets
combined_dataset = audio_dataset.train_test_split(test_size=0.1, seed=42)["train"]
combined_dataset = combined_dataset.add_column(translation_dataset["translation"], "translation")

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=100,
)

# Define the Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=combined_dataset,
    tokenizer=translation_tokenizer,
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("./helsinki_model")
