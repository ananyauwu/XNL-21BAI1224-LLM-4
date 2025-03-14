import os
import logging
import requests
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import re
import time
app = Flask(__name__)
CORS(app)

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

