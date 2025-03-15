import os
import logging
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import spacy
import pytesseract
from PIL import Image
import nltk
import numpy as np
from datasets import load_dataset, Dataset
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

# Define a function to handle null values/missing values in the dataset
def handle_missing_values(examples):
    for key in examples.keys():
        if isinstance(examples[key], list):
            examples[key] = [value if value is not None else "N/A" for value in examples[key]]
        else:
            examples[key] = examples[key] if examples[key] is not None else "N/A"
    return examples

# Acquire the training data from Hugging Face
DATA_NAME = "financial_phrasebank"
CONFIG_NAME = "sentences_allagree"
financial_dataset = load_dataset(DATA_NAME, CONFIG_NAME)

# Apply the function to handle missing values
financial_dataset = financial_dataset.map(handle_missing_values)

# Print sample data from the dataset
print("Sample data from the dataset:", financial_dataset['train'][0])

# Split the financial dataset into training and testing datasets
train_size = int(0.7 * len(financial_dataset['train']))
train_dataset = financial_dataset['train'].select(range(train_size))
test_dataset = financial_dataset['train'].select(range(train_size, len(financial_dataset['train'])))

# We prefix our tasks with "answer the question"
prefix = "Please answer this question: "

# Define the preprocessing function
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["sentence"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    
    # Convert labels to strings and then tokenize
    labels = [str(label) for label in examples["label"]]  # Convert labels to strings
    labels = tokenizer(text_target=labels, max_length=512, truncation=True)
    
    # Ensure labels are lists of integers
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Map the preprocessing function across our dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

print("After preprocessing:", train_dataset[0]["labels"])

# Ensure labels are properly formatted
def format_labels(examples):
    examples["labels"] = [[label] if isinstance(label, int) else label for label in examples["labels"]]
    return examples

train_dataset = train_dataset.map(format_labels, batched=True)
test_dataset = test_dataset.map(format_labels, batched=True)

# Print sample labels to debug
print("Sample labels from train_dataset:", train_dataset[0]["labels"])
print("Sample labels from test_dataset:", test_dataset[0]["labels"])

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
    eval_strategy="epoch",  # Use eval_strategy instead of evaluation_strategy
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