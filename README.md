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
