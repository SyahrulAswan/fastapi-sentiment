from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and tokenizer
logging.info("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")
model = AutoModelForSequenceClassification.from_pretrained("w11wo/indonesian-roberta-base-sentiment-classifier")
logging.info("Model and tokenizer loaded successfully.")

# Define sentiment mapping
sentiment_mapping = {0: 1, 1: 0, 2: -1}  # Assuming model output follows: 0=pos, 1=neutral, 2=neg

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str

@app.on_event("startup")
def startup_event():
    logging.info("API is starting...")

@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    if not request.text:
        logging.warning("Received empty text input.")
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    logging.info(f"Received text: {request.text}")
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    sentiment_class = torch.argmax(outputs.logits, dim=1).item()
    sentiment = sentiment_mapping[sentiment_class]
    
    logging.info(f"Predicted sentiment: {sentiment}")
    return {"sentiment": sentiment}

@app.on_event("shutdown")
def shutdown_event():
    logging.info("API is shutting down...")
