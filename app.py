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

app = Flask(__name__)
CORS(app)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

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

api_url = get_env_variable('LLM_API_URL')
api_key = get_env_variable('LLM_API_KEY')

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

# Endpoint to handle chatbot queries
@app.route('/chatbot', methods=['POST'])
@cross_origin()
def chatbot():
    user_input = request.json.get('question')
    if not user_input:
        return jsonify({'error': 'No question provided'}), 400

    # Call the LLM model API to generate a response
    api_url = get_env_variable('LLM_API_URL')
    api_key = get_env_variable('LLM_API_KEY')
    headers = {'Authorization': f'Bearer {api_key}'}
    payload = {'prompt': user_input, 'max_tokens': 150}

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        answer = response.json().get('choices', [{}])[0].get('text', '').strip()
        return jsonify({'answer': answer})
    except requests.RequestException as e:
        logging.error(f"Error calling LLM API: {e}")
        return jsonify({'answer': 'This is a standard response for testing purposes.'}), 500

if __name__ == '__main__':
    app.run(debug=True)

