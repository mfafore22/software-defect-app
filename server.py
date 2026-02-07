from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch

app = Flask(__name__)
CORS(app)

# ============================================
# LOAD MODEL
# ============================================
print("Loading model...")

base_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
model = PeftModel.from_pretrained(base_model, "./software-defect-model")
tokenizer = AutoTokenizer.from_pretrained("./software-defect-model")
model.eval()

print("Model loaded!")

# ============================================
# API ROUTES
# ============================================

@app.route("/")
def home():
    return jsonify({
        "message": "Software Defect Classifier API",
        "endpoints": {
            "POST /api/predict": "Predict if code has defects",
            "GET /api/health": "Health check"
        }
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    
    # Get metrics
    loc = data.get("loc", 0)
    complexity = data.get("complexity", 0)
    essential = data.get("essential", 0)
    integration = data.get("integration", 0)
    
    # Create text
    text = f"This software module has {loc} lines of code, cyclomatic complexity {complexity}, essential complexity {essential}, and integration complexity {integration}."
    
    # Tokenize
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding="max_length")
    
    # FIXED: Only send input_ids and attention_mask (DistilBERT doesn't use token_type_ids)
    inputs = {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"]
    }
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = outputs.logits.argmax(dim=-1).item()
    confidence = probs.max().item()
    
    return jsonify({
        "prediction": "Defect" if pred == 1 else "No Defect",
        "confidence": round(confidence * 100, 2),
        "input_text": text
    })

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)