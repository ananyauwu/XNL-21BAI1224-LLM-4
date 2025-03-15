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
from transformers import Trainer

app = Flask(__name__)
CORS(app)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the tokenizer, model, and data collator
MODEL_NAME = "google/flan-t5-small"
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

# Load the finqa dataset from Hugging Face
ds = load_dataset("bilalRahib/fiqa-personal-finance-dataset")

# Apply the function to remove missing values
ds = ds.filter(lambda x: all(v is not None for v in x.values()))

# Print the first 10 lines from the dataset
print("First 10 lines from the dataset:")
for i in range(10):
    print(ds['train'][i])

# Print sample data from the dataset
print("Sample data from the dataset:", ds['train'][0])

# Split the financial dataset into training, testing, and validation datasets
train_size = int(0.7 * len(ds['train']))
val_size = int(0.15 * len(ds['train']))
train_dataset = ds['train'].select(range(train_size))
val_dataset = ds['train'].select(range(train_size, train_size + val_size))
test_dataset = ds['train'].select(range(train_size + val_size, len(ds['train'])))

# We prefix our tasks with "answer the question"
prefix = "Please answer this financial question: "

# Define the preprocessing function
def preprocess_function(examples):
    # Combine the input and output into a single input string
    inputs = [prefix + inp for inp in examples["input"]]
    
    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    # Tokenize outputs (target labels)
    labels = tokenizer(examples["output"], max_length=128, truncation=True, padding="max_length", return_tensors="pt")
    
    # Add labels to the model inputs
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Apply the preprocessing function to the dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

print("After preprocessing:", train_dataset[0])

# Download NLTK data
nltk.download("punkt", quiet=True)

# Define the compute_metrics function
def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Calculate ROUGE score
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
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-4,
    per_device_train_batch_size=4,  # Reduce batch size for CPU
    per_device_eval_batch_size=2,   # Reduce batch size for CPU
    gradient_accumulation_steps=2,  # Use gradient accumulation
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    use_cpu=True  # Use CPU instead of no_cuda
)

class DebugTrainer(Trainer):
    def training_step(self, model, inputs):
        try:
            return super().training_step(model, inputs)
        except Exception as e:
            print("Error during training step:", e)
            print("Inputs:", inputs)
            raise e

# Use the DebugTrainer instead of the default Trainer
trainer = DebugTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
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
    inputs = tokenizer("Please answer this financial question: " + user_input, return_tensors="pt")
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)