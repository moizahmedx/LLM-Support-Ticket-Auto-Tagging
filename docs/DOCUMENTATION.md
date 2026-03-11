# Technical Documentation

Comprehensive technical documentation for the Support Ticket Auto-Tagging system.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Documentation](#module-documentation)
3. [Model Details](#model-details)
4. [API Reference](#api-reference)
5. [Performance Optimization](#performance-optimization)
6. [Deployment Guide](#deployment-guide)

## Architecture Overview

### System Design

The project follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    Data Pipeline                         │
│  Dataset Download → Preprocessing → Train/Val/Test Split │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                 Classification Models                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Zero-Shot   │  │  Few-Shot    │  │  Fine-Tuned  │  │
│  │  Classifier  │  │  Classifier  │  │    Model     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              Evaluation & Prediction                     │
│  Metrics Calculation → Visualization → Inference API     │
└─────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Scalability**: Can handle datasets of varying sizes
3. **Flexibility**: Easy to swap models or add new approaches
4. **Production-Ready**: Includes error handling, logging, and validation

## Module Documentation

### 1. dataset_downloader.py

**Purpose**: Downloads support ticket datasets from HuggingFace.

**Key Classes**:
- `DatasetDownloader`: Handles dataset retrieval and storage

**Key Methods**:
```python
download() -> pd.DataFrame
    Downloads dataset and saves to data/raw/
    Returns: DataFrame with ticket data
```

**Configuration**:
- Default dataset: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
- Fallback dataset: `banking77`
- Output: `data/raw/support_tickets.csv`

### 2. data_preprocessing.py

**Purpose**: Cleans and prepares data for training.

**Key Classes**:
- `DataPreprocessor`: Handles all preprocessing operations

**Key Methods**:
```python
clean_text(text: str) -> str
    Normalizes text (lowercase, remove URLs, special chars)
    
preprocess(df: pd.DataFrame) -> pd.DataFrame
    Applies full preprocessing pipeline
    
split_data(df, test_size=0.2, val_size=0.1) -> tuple
    Splits data into train/val/test sets
```

**Preprocessing Steps**:
1. Text normalization (lowercase)
2. URL and email removal
3. Special character cleaning
4. Duplicate removal
5. Missing value handling
6. Stratified splitting

### 3. zero_shot_classifier.py

**Purpose**: Classifies tickets without training using pre-trained models.

**Key Classes**:
- `ZeroShotClassifier`: Zero-shot classification implementation

**Model**: `facebook/bart-large-mnli`

**Key Methods**:
```python
predict_single(text: str, top_k: int = 3) -> list
    Predicts top K categories for one ticket
    Returns: [(category, confidence), ...]
    
predict_batch(texts: list, top_k: int = 3) -> list
    Batch prediction for multiple tickets
```

**Advantages**:
- No training required
- Works with any category set
- Good for prototyping

**Limitations**:
- Slower inference
- Lower accuracy than fine-tuned models
- Limited to ~50 categories for efficiency

### 4. few_shot_classifier.py

**Purpose**: Improves zero-shot with example-based learning.

**Key Classes**:
- `FewShotClassifier`: Few-shot learning implementation

**Key Methods**:
```python
load_examples(train_data_path: str)
    Loads example tickets for each category
    
create_prompt(text: str, category: str) -> str
    Creates enhanced prompt with examples
    
predict_single(text: str, top_k: int = 3) -> list
    Predicts using few-shot approach
```

**Configuration**:
- `n_examples`: Number of examples per category (default: 3)
- Uses same base model as zero-shot

**Advantages**:
- Better accuracy than zero-shot
- Minimal training data needed
- Flexible category adaptation

### 5. fine_tuning.py

**Purpose**: Fine-tunes a transformer model for optimal performance.

**Key Classes**:
- `TicketClassifierTrainer`: Handles model training

**Model**: `distilbert-base-uncased` (can be changed)

**Key Methods**:
```python
load_data() -> tuple
    Loads and prepares datasets
    Returns: (train_dataset, val_dataset, test_dataset)
    
initialize_model()
    Initializes model with correct number of labels
    
train(train_dataset, val_dataset, output_dir: str)
    Fine-tunes the model
    
compute_metrics(eval_pred) -> dict
    Calculates accuracy, F1, precision, recall
```

**Training Configuration**:
```python
learning_rate = 2e-5
batch_size = 16
num_epochs = 3
max_length = 128
optimizer = AdamW
```

**Output**:
- Trained model: `models/fine_tuned/`
- Label mappings: `models/fine_tuned/label_mappings.json`

### 6. evaluation.py

**Purpose**: Compares all models and generates reports.

**Key Classes**:
- `ModelEvaluator`: Evaluation and visualization

**Key Methods**:
```python
load_results() -> dict
    Loads results from all models
    
plot_accuracy_comparison(results: dict)
    Creates accuracy comparison plot
    
plot_metrics_comparison(results: dict)
    Creates detailed metrics plot
    
generate_summary_report(results: dict)
    Creates text summary report
```

**Outputs**:
- `results/accuracy_comparison.png`
- `results/detailed_metrics.png`
- `results/evaluation_report.txt`

### 7. predict.py

**Purpose**: Production inference interface.

**Key Classes**:
- `TicketPredictor`: Unified prediction interface

**Key Methods**:
```python
predict(ticket_text: str, top_k: int = 3) -> list
    Predicts top K tags for a ticket
    Returns: [(tag, confidence), ...]
    
predict_batch(ticket_texts: list, top_k: int = 3) -> list
    Batch prediction for multiple tickets
```

**Usage**:
```python
from src.predict import TicketPredictor

predictor = TicketPredictor()
tags = predictor.predict("My laptop won't turn on")
```

## Model Details

### Zero-Shot Model

**Architecture**: BART (Bidirectional and Auto-Regressive Transformers)
- Model: `facebook/bart-large-mnli`
- Parameters: 406M
- Training: Natural Language Inference (MNLI dataset)

**How it works**:
1. Converts classification to entailment task
2. Checks if ticket text entails each category
3. Returns categories with highest entailment scores

### Few-Shot Model

**Architecture**: Same as zero-shot with enhanced prompting
- Adds example tickets to context
- Improves category understanding
- No parameter updates

### Fine-Tuned Model

**Architecture**: DistilBERT
- Model: `distilbert-base-uncased`
- Parameters: 66M (smaller, faster than BERT)
- Training: Custom classification head

**Architecture Details**:
```
Input Text
    ↓
Tokenizer (WordPiece)
    ↓
DistilBERT Encoder (6 layers)
    ↓
[CLS] Token Representation
    ↓
Classification Head (Linear + Softmax)
    ↓
Category Probabilities
```

## API Reference

### TicketPredictor Class

```python
class TicketPredictor:
    def __init__(self, model_path="models/fine_tuned", use_zero_shot=False):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to fine-tuned model
            use_zero_shot: Use zero-shot instead of fine-tuned
        """
        
    def predict(self, ticket_text: str, top_k: int = 3) -> list:
        """
        Predict tags for a ticket.
        
        Args:
            ticket_text: Support ticket text
            top_k: Number of top predictions
            
        Returns:
            List of (tag, confidence) tuples
            
        Example:
            >>> predictor = TicketPredictor()
            >>> tags = predictor.predict("Password reset not working")
            >>> print(tags)
            [('account', 0.89), ('password_reset', 0.76), ('login', 0.45)]
        """
```

## Performance Optimization

### Speed Optimization

1. **Use GPU**:
```python
# Change device=-1 to device=0 in classifier files
self.classifier = pipeline("zero-shot-classification", device=0)
```

2. **Batch Processing**:
```python
# Process multiple tickets at once
predictions = predictor.predict_batch(ticket_list)
```

3. **Model Quantization**:
```python
# Use quantized models for faster inference
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    torch_dtype=torch.float16  # Half precision
)
```

### Memory Optimization

1. **Reduce Batch Size**:
```python
# In fine_tuning.py
per_device_train_batch_size = 8  # Reduce from 16
```

2. **Reduce Max Length**:
```python
# In fine_tuning.py
max_length = 64  # Reduce from 128
```

3. **Use Gradient Accumulation**:
```python
# In TrainingArguments
gradient_accumulation_steps = 2
```

## Deployment Guide

### Option 1: REST API with Flask

```python
from flask import Flask, request, jsonify
from src.predict import TicketPredictor

app = Flask(__name__)
predictor = TicketPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticket_text = data.get('ticket_text', '')
    top_k = data.get('top_k', 3)
    
    predictions = predictor.predict(ticket_text, top_k)
    
    return jsonify({
        'ticket': ticket_text,
        'predictions': [
            {'tag': tag, 'confidence': float(conf)}
            for tag, conf in predictions
        ]
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Option 2: FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import TicketPredictor

app = FastAPI()
predictor = TicketPredictor()

class TicketRequest(BaseModel):
    ticket_text: str
    top_k: int = 3

@app.post("/predict")
async def predict(request: TicketRequest):
    predictions = predictor.predict(request.ticket_text, request.top_k)
    return {
        'predictions': [
            {'tag': tag, 'confidence': conf}
            for tag, conf in predictions
        ]
    }
```

### Option 3: Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/predict.py"]
```

### Production Considerations

1. **Model Versioning**: Track model versions and performance
2. **Monitoring**: Log predictions and confidence scores
3. **A/B Testing**: Compare model versions in production
4. **Caching**: Cache predictions for common tickets
5. **Rate Limiting**: Prevent API abuse
6. **Error Handling**: Graceful degradation on failures

## Best Practices

1. **Regular Retraining**: Retrain models with new data monthly
2. **Data Quality**: Monitor input data quality
3. **Confidence Thresholds**: Set minimum confidence for auto-tagging
4. **Human Review**: Flag low-confidence predictions for review
5. **Feedback Loop**: Collect corrections to improve model

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size
   - Use CPU instead of GPU
   - Use smaller model

2. **Slow Inference**:
   - Use fine-tuned model instead of zero-shot
   - Enable GPU
   - Use model quantization

3. **Low Accuracy**:
   - Collect more training data
   - Increase training epochs
   - Try different models
   - Adjust learning rate

## References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [BART Paper](https://arxiv.org/abs/1910.13461)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Zero-Shot Learning](https://arxiv.org/abs/1909.00161)
