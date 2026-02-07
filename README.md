# Software Defect Classifier

A web application that predicts whether a software module contains defects based on code metrics. Built using a fine-tuned DistilBERT model with LoRA.

## What It Does

This app takes software metrics (lines of code, complexity measures) and predicts if the code is likely to have bugs. The model was trained on the JM1 dataset from NASA, which contains metrics from real software projects.

Input: Code metrics like lines of code, cyclomatic complexity
Output: "Defect" or "No Defect" with confidence score

## Tech Stack

- Model: DistilBERT with LoRA (Parameter-Efficient Fine-Tuning)
- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- Training: Google Colab

## Dataset

JM1 Software Defect Dataset from NASA/PROMISE Repository.

Source: https://www.kaggle.com/datasets/semustafacevik/software-defect-prediction

### What is JM1?

JM1 is a dataset from NASA containing information about real software code from NASA projects. Each entry is like a report card for a piece of code.

### What the metrics mean:

| Metric | Simple Meaning |
|--------|----------------|
| Lines of Code (LOC) | How long is the code? |
| Cyclomatic Complexity | How many decisions/branches does the code make? |
| Essential Complexity | How tangled/messy is the code? |
| Integration Complexity | How connected is the code to other parts? |
| Defects (Yes/No) | Did this code have bugs? |

### General patterns in the data:

- Short, simple code is less likely to have bugs
- Long, complex code is more likely to have bugs

The full dataset contains 10,885 software modules with 21 features.

## How the Model Learns

### What happened during training:

The model saw 400 examples like:

```
"This code has 100 lines, complexity 10..." → No bugs
"This code has 500 lines, complexity 45..." → Bugs
"This code has 50 lines, complexity 5..." → No bugs
```

After seeing these examples, the model learned patterns between code metrics and defects.

### Training outcome:

The model now makes educated guesses based on the patterns it learned. It is correct about 70-75% of the time. It works best with normal values similar to what it saw during training.

## Project Structure

```
software-defect-app/
├── server.py
├── requirements.txt
├── README.md
├── software-defect-model/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
└── frontend/
    └── index.html
```

## How to Run

### Step 1: Clone or download this project

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Start the backend

```bash
python server.py
```

Wait until you see "Model loaded!" and "Running on http://127.0.0.1:5000"

Keep this terminal open. The backend must stay running.

### Step 4: Open the frontend

You have two options:

**Option A: Double-click the file**
1. Open File Explorer
2. Go to the `frontend` folder
3. Double-click `index.html`
4. It opens in your browser

**Option B: Use Live Server in VS Code**
1. Install the "Live Server" extension by Ritwick Dey in VS Code
2. Right-click on `frontend/index.html`
3. Click "Open with Live Server"
4. Browser opens automatically at `http://127.0.0.1:5500`

### Step 5: Use the app

Enter code metrics and click "Analyze Code" to get a prediction.

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "Make sure backend is running" | Check terminal shows "Running on http://127.0.0.1:5000" |
| Frontend not loading | Try Option A (double-click) instead of Live Server |
| Model loading slow | First load takes 30-60 seconds, be patient |
| CORS error in console | Make sure flask-cors is installed |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | API info |
| /api/predict | POST | Predict defect |
| /api/health | GET | Health check |

### Example Request

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"loc": 100, "complexity": 10, "essential": 5, "integration": 3}'
```

### Example Response

```json
{
  "prediction": "No Defect",
  "confidence": 86.85,
  "input_text": "This software module has 100 lines of code..."
}
```

## Training Details

- Base Model: distilbert-base-uncased
- Fine-tuning: LoRA (rank=8, alpha=32)
- Training Samples: 400
- Test Samples: 100
- Epochs: 3
- Learning Rate: 5e-5
- Batch Size: 8
- Weight Decay: 0.01

### Training Results

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1 | 0.465669 | 0.525127 |
| 2 | 0.478681 | 0.521476 |
| 3 | 0.487545 | 0.523041 |

## Requirements

- Python 3.8+
- Flask
- Transformers
- PyTorch
- PEFT

## Why DistilBERT?

We used DistilBERT instead of BERT for this project.

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| BERT | 110M parameters | Slower | 100% |
| DistilBERT | 66M parameters | 60% faster | 97% |

DistilBERT is a smaller, faster version of BERT. It runs on free Google Colab GPUs and regular laptops without needing expensive hardware. For a learning project, it gives good results without the heavy requirements.

### Technical Note

DistilBERT does not use `token_type_ids` like BERT does. If you see an error about "unexpected keyword argument token_type_ids", the fix is to only pass `input_ids` and `attention_mask` to the model:

```python
inputs = {
    "input_ids": tokens["input_ids"],
    "attention_mask": tokens["attention_mask"]
}
outputs = model(**inputs)
```

## Why LoRA Instead of Full Fine-Tuning?

### The Cost Problem

Full fine-tuning means updating all 66 million parameters in the model. This requires:

| Resource | Full Fine-Tuning | LoRA |
|----------|------------------|------|
| GPU Memory | 16GB+ | 4GB |
| Training Time | Hours | Minutes |
| Cost (Cloud GPU) | $10-50 | Free (Colab) |
| Storage | 250MB+ | 5MB |

### What LoRA Does

LoRA (Low-Rank Adaptation) freezes the original model and only trains small adapter layers.

```
Full Fine-Tuning: Update 66,000,000 parameters
LoRA: Update only 600,000 parameters (less than 1%)
```

This makes training possible on free resources like Google Colab.

### Tradeoffs

| Approach | Pros | Cons |
|----------|------|------|
| Full Fine-Tuning | Best accuracy, full control | Expensive, slow, needs big GPU |
| LoRA | Cheap, fast, runs anywhere | Slightly less accurate |

For learning and small projects, LoRA is the practical choice.

## Why Extreme Values Do Not Work

### The 1 Million Lines Problem

If you enter 1,000,000 lines of code, the model still says "No Defect". This is wrong, but the model cannot know that.

### Why This Happens

The model only learned from values in the training data:

```
Training data range:
- Lines of Code: 1 to 500
- Complexity: 1 to 50

Your input:
- Lines of Code: 1,000,000 (never seen before)
```

The model converts your input to text: "This software module has 1000000 lines of code..."

But during training, it never saw text with numbers that large. It does not understand what 1,000,000 means. It just sees an unfamiliar pattern and guesses.

### How to Fix This

To handle extreme values, you would need to:

1. Include extreme values in training data
2. Add input validation to reject unrealistic numbers
3. Normalize inputs (convert all numbers to a 0-1 scale)
4. Train on much more data with wider range

For this project, we kept it simple and expect users to enter realistic values.

## Limitations

- Trained on limited data (500 samples)
- Only uses 4 metrics for prediction
- Designed for educational purposes
- Model only works well within the range of values it was trained on

### Input Range

The model was trained on normal software metric values. Using extreme values (like 1 million) will produce unreliable predictions because the model has never seen such numbers during training.

Recommended input ranges:

| Metric | Realistic Range |
|--------|-----------------|
| Lines of Code | 10 - 1000 |
| Cyclomatic Complexity | 1 - 50 |
| Essential Complexity | 1 - 30 |
| Integration Complexity | 1 - 20 |

Values outside these ranges may still return a prediction, but the result should not be trusted.

### Other Edge Cases

| Input | Behavior |
|-------|----------|
| Negative numbers | May produce unpredictable results |
| Zero for all fields | Will return a prediction but not meaningful |
| Very large numbers | Model guesses incorrectly (not trained on these) |
| Empty fields | Frontend uses default value of 0 |

## How to Scale This Application

This app runs locally on one computer. To make it available to many users, you need to deploy it.

### Current Setup (Local)

```
Your Computer:
- Backend (Flask) on port 5000
- Frontend (HTML) on port 5500
- Only you can access it
```

### Production Setup (Deployed)

```
Cloud Server:
- Backend running 24/7
- Frontend hosted on CDN
- Anyone with the URL can use it
```

### Deployment Options

| Platform | Cost | Best For |
|----------|------|----------|
| Hugging Face Spaces | Free | ML models (recommended) |
| Render | Free tier | Full apps |
| Railway | Free tier | Full apps |
| AWS/GCP/Azure | Paid | Large scale, enterprise |

### Steps to Scale

1. Replace Flask dev server with production server (Gunicorn)
2. Add a requirements.txt with all dependencies
3. Create a Dockerfile or use platform-specific config
4. Deploy backend to cloud platform
5. Host frontend on Vercel, Netlify, or same platform
6. Update frontend to call the cloud backend URL

### Hardware Requirements for Scale

| Users | Recommended Setup |
|-------|-------------------|
| 1-10 | Local or free tier |
| 10-100 | Small cloud instance (2GB RAM) |
| 100-1000 | Dedicated GPU instance |
| 1000+ | Load balancer + multiple instances |

### Cost Estimates

| Scale | Monthly Cost |
|-------|--------------|
| Learning/Demo | Free |
| Small team | $5-20 |
| Production | $50-200 |
| Enterprise | $500+ |

## Author

Fafore Michael Oluwaseun