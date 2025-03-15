import os
import logging
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import re
import time
import spacy
import pytesseract
from PIL import Image
import nltk
import numpy as np
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from rouge import Rouge

app = Flask(__name__)
CORS(app)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the tokenizer, model, and data collator
MODEL_NAME = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Logging configuration
logging.basicConfig(level=logging.INFO)

# Helper function to load environment variables
def get_env_variable(var_name, default=None):
    value = os.getenv(var_name, default)
    if value is None:
        logging.error(f"Missing environment variable: {var_name}")
        raise EnvironmentError(f"Required environment variable {var_name} is not set.")
    logging.info(f"Loaded environment variable: {var_name}")
    return value

@app.route('/')
def index():
    return app.send_static_file('interface.html')

# Textual data preprocessing
def preprocess_text(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    sentiment = doc.sentiment
    return {"entities": entities, "keywords": keywords, "sentiment": sentiment}

# OCR for image data
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Parse tabular data
def parse_tabular_data(file_path):
    df = pd.read_csv(file_path)
    return df.to_dict()

# Acquire the training data from Hugging Face
DATA_NAME = "financial_phrasebank"
CONFIG_NAME = "sentences_allagree"
financial_dataset = load_dataset(DATA_NAME, CONFIG_NAME)

# Fetch news data from NewsAPI
def fetch_news_data(query, from_date):
    url = (f'http://newsapi.org/v2/everything?q={query}&from={from_date}&sortBy=popularity&apiKey=ce1ff6d5196d4f7d990304d44713d946')
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        logging.error(f"Error fetching news data: {response.status_code}")
        return []

news_data = fetch_news_data('finance', '2023-01-01')

# Combine financial dataset and news data
def combine_datasets(financial_dataset, news_data):
    news_sentences = [article['description'] for article in news_data if article['description']]
    news_labels = ['neutral'] * len(news_sentences)  # Assuming neutral sentiment for simplicity
    combined_sentences = financial_dataset['train']['sentence'] + news_sentences
    combined_labels = financial_dataset['train']['label'] + news_labels
    return {'sentence': combined_sentences, 'label': combined_labels}

combined_data = combine_datasets(financial_dataset, news_data)

# Split the combined data into training and testing datasets
train_size = int(0.7 * len(combined_data['sentence']))
train_dataset = {'sentence': combined_data['sentence'][:train_size], 'label': combined_data['label'][:train_size]}
test_dataset = {'sentence': combined_data['sentence'][train_size:], 'label': combined_data['label'][train_size:]}

# We prefix our tasks with "answer the question"
prefix = "Please answer this question: "

# Define the preprocessing function
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["sentence"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = [str(label) for label in examples["label"]]  # Convert labels to strings
    labels = tokenizer(text_target=labels, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Map the preprocessing function across our dataset
train_dataset = preprocess_function(train_dataset)
test_dataset = preprocess_function(test_dataset)

# Download NLTK data
nltk.download("punkt", quiet=True)

# Define the compute_metrics function
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE expects newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    rouge = Rouge()
    result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    return result

class SaveBestModelCallback(TrainerCallback):
    def __init__(self):
        self.best_metric = None
        self.best_model_path = None

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs.get("metrics")
        if metrics:
            current_metric = metrics.get("eval_loss")
            if self.best_metric is None or current_metric < self.best_metric:
                self.best_metric = current_metric
                self.best_model_path = os.path.join(args.output_dir, "best_model")
                kwargs["model"].save_pretrained(self.best_model_path)
                kwargs["tokenizer"].save_pretrained(self.best_model_path)
                logging.info(f"New best model saved with eval_loss: {self.best_metric}")

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Ensure save strategy matches evaluation strategy
    learning_rate=3e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)

# Set up the trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[SaveBestModelCallback()]
)

# Fine-tune the model
trainer.train()

# Endpoint to handle chatbot queries
@app.route('/chatbot', methods=['POST'])
@cross_origin()
def chatbot():
    user_input = request.json.get('question')
    if not user_input:
        return jsonify({'error': 'No question provided'}), 400

    # Generate response using the fine-tuned model
    inputs = tokenizer("Please answer this question: " + user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)

