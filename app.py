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

# Define a function to remove sets with missing values in the dataset
def remove_missing_values(examples):
    for key in examples.keys():
        if isinstance(examples[key], list):
            examples[key] = [value for value in examples[key] if value is not None]
        else:
            if examples[key] is None:
                return None
    return examples

# Acquire the training data from Hugging Face
DATA_NAME = "financial_phrasebank"
CONFIG_NAME = "sentences_allagree"
financial_dataset = load_dataset(DATA_NAME, CONFIG_NAME)

# Apply the function to remove missing values
financial_dataset = financial_dataset.filter(lambda x: all(v is not None for v in x.values()))

# Print the first 20 lines from the dataset
print("First 20 lines from the dataset:")
for i in range(20):
    print(financial_dataset['train'][i])

# Print sample data from the dataset
print("Sample data from the dataset:", financial_dataset['train'][0])

# Split the financial dataset into training and testing datasets
train_size = int(0.7 * len(financial_dataset['train']))
train_dataset = financial_dataset['train'].select(range(train_size))
test_dataset = financial_dataset['train'].select(range(train_size, len(financial_dataset['train'])))

# We prefix our tasks with "answer the question"
prefix = "Please answer this question: "

# Define the preprocessing function to ignore labels and take sentences only
def preprocess_function(examples):
    # Use the sentences as both inputs and labels (for language modeling)
    inputs = [prefix + doc for doc in examples["sentence"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    
    # Use the same input as the target (labels) for language modeling
    model_inputs["labels"] = model_inputs["input_ids"]  # Set labels = input_ids
    
    return model_inputs

print("After preprocessing:", train_dataset[0])

# Download NLTK data
nltk.download("punkt", quiet=True)

# Define the compute_metrics function
def compute_metrics(eval_preds):
    preds, _ = eval_preds  # Ignore labels

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Calculate average length of generated text
    avg_length = np.mean([len(pred.split()) for pred in decoded_preds])

    # Calculate diversity (ratio of unique words to total words)
    unique_words = set()
    total_words = 0
    for pred in decoded_preds:
        words = pred.split()
        unique_words.update(words)
        total_words += len(words)
    diversity = len(unique_words) / total_words if total_words > 0 else 0

    return {"avg_length": avg_length, "diversity": diversity}

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
    evaluation_strategy="epoch",  # Use evaluation_strategy instead of eval_strategy
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