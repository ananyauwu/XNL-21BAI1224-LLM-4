# XNL-21BAI1224-LLM-4
Overview

This project is a chatbot hosted on a Virtual Machine (VM) using Google Cloud. The chatbot utilizes Natural Language Processing (NLP) techniques, including the T5 transformer model for text generation, to provide responses based on user input. The application is built using Flask and integrates various libraries for data processing, OCR, and sentiment analysis.

Features

NLP Processing: Utilizes spaCy for named entity recognition and text processing.

OCR Support: Uses Tesseract OCR to extract text from images.

Tabular Data Parsing: Reads and processes CSV files using pandas.

Text Generation: Fine-tunes the FLAN-T5 model for generating answers.

API Interface: Provides a REST API using Flask for chatbot interactions.

Model Training: Includes training and evaluation of the FLAN-T5 model with datasets from Hugging Face.

Hosted on Google Cloud: Deployed on a VM for accessibility and scalability.

Prerequisites

Ensure that you have the following installed:

Python 3.8+

Flask

Requests

Pandas

spaCy

pytesseract

PIL (Pillow)

NLTK

NumPy

Datasets (Hugging Face)

Transformers (Hugging Face)

ROUGE (for evaluation metrics)

Installation

Clone the repository:

git clone https://github.com/your-repo/chatbot.git
cd chatbot

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Install dependencies:

pip install -r requirements.txt

Download the required NLP models:

python -m spacy download en_core_web_sm

Set up environment variables as needed.

Usage

Running the Application

To start the Flask application, run:

python app.py

The application will be available at http://localhost:5000/.

API Endpoints

1. Chatbot Interaction

Endpoint: /chatbot

Method: POST

Request Body:

{
  "question": "What is AI?"
}

Response:

{
  "answer": "AI stands for Artificial Intelligence."
}

2. Text Preprocessing

Function: Extracts keywords, entities, and sentiment from input text.

Usage: Called internally for NLP tasks.

3. OCR for Image Data

Function: Extracts text from images using Tesseract OCR.

Usage: Used in cases where text needs to be extracted from uploaded images.

Model Training

To fine-tune the FLAN-T5 model on financial phrase datasets, the script automates:

Data loading and preprocessing

Model training and evaluation

Best model saving using callbacks

Deployment

The chatbot is hosted on a Google Cloud VM. Ensure the necessary firewall rules are configured to allow traffic to the Flask application.

Deploying to Google Cloud VM

Set up a VM instance in Google Cloud.

Install Python and dependencies on the VM.

Run the application in the background using:

nohup python app.py &

Configure a reverse proxy (e.g., Nginx) if required.
