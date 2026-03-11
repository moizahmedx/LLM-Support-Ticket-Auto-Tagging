# Deployment Guide

Complete guide for deploying the Support Ticket Auto-Tagging system to production.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Local Deployment](#local-deployment)
3. [REST API Deployment](#rest-api-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Monitoring](#monitoring)
7. [Scaling](#scaling)

---

## Prerequisites

- Python 3.8+
- Trained model (run fine-tuning first)
- 4GB+ RAM
- (Optional) GPU for faster inference

---

## Local Deployment

### Step 1: Train the Model
```bash
python src/dataset_downloader.py
python src/data_preprocessing.py
python src/fine_tuning.py
```

### Step 2: Test Predictions
```bash
python src/predict.py
```

### Step 3: Use in Your Application
```python
from src.predict import TicketPredictor

predictor = TicketPredictor()
tags = predictor.predict("Your ticket text here")
```

---

## REST API Deployment

### Option 1: Flask API

Create `api_flask.py`:
```python
from flask import Flask, request, jsonify
from src.predict import TicketPredictor

app = Flask(__name__)
predictor = TicketPredictor()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticket_text = data.get('ticket_text', '')
    top_k = data.get('top_k', 3)
    
    if not ticket_text:
        return jsonify({'error': 'ticket_text required'}), 400
    
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

Run:
```bash
pip install flask
python api_flask.py
```

Test:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"ticket_text": "My laptop won'\''t turn on"}'
```

### Option 2: FastAPI

Create `api_fastapi.py`:
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import TicketPredictor

app = FastAPI(title="Support Ticket Auto-Tagging API")
predictor = TicketPredictor()

class TicketRequest(BaseModel):
    ticket_text: str
    top_k: int = 3

class PredictionResponse(BaseModel):
    ticket: str
    predictions: list

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TicketRequest):
    if not request.ticket_text:
        raise HTTPException(status_code=400, 
                          detail="ticket_text required")
    
    predictions = predictor.predict(request.ticket_text, 
                                   request.top_k)
    
    return {
        "ticket": request.ticket_text,
        "predictions": [
            {"tag": tag, "confidence": conf}
            for tag, conf in predictions
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Run:
```bash
pip install fastapi uvicorn
python api_fastapi.py
```

Access docs: http://localhost:8000/docs

---

## Docker Deployment

### Create Dockerfile

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "api_fastapi.py"]
```

### Build and Run
```bash
# Build image
docker build -t ticket-tagger:latest .

# Run container
docker run -p 8000:8000 ticket-tagger:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 ticket-tagger:latest
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/fine_tuned
    volumes:
      - ./models:/app/models
    restart: unless-stopped
```

Run:
```bash
docker-compose up -d
```

---

## Cloud Deployment

### AWS Deployment

#### Using EC2
1. Launch EC2 instance (t3.medium or larger)
2. Install dependencies
3. Copy model files
4. Run API server
5. Configure security groups (port 8000)

#### Using ECS/Fargate
1. Push Docker image to ECR
2. Create ECS task definition
3. Create ECS service
4. Configure load balancer

#### Using Lambda
For serverless deployment (cold start considerations):
```python
import json
from src.predict import TicketPredictor

predictor = None

def lambda_handler(event, context):
    global predictor
    
    if predictor is None:
        predictor = TicketPredictor()
    
    body = json.loads(event['body'])
    ticket_text = body.get('ticket_text', '')
    
    predictions = predictor.predict(ticket_text)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'predictions': [
                {'tag': tag, 'confidence': float(conf)}
                for tag, conf in predictions
            ]
        })
    }
```

### Google Cloud Platform

#### Using Cloud Run
```bash
# Build and push
gcloud builds submit --tag gcr.io/PROJECT_ID/ticket-tagger

# Deploy
gcloud run deploy ticket-tagger \
  --image gcr.io/PROJECT_ID/ticket-tagger \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure

#### Using Azure Container Instances
```bash
# Create resource group
az group create --name ticket-tagger-rg --location eastus

# Deploy container
az container create \
  --resource-group ticket-tagger-rg \
  --name ticket-tagger \
  --image your-registry/ticket-tagger:latest \
  --dns-name-label ticket-tagger \
  --ports 8000
```

---

## Monitoring

### Logging

Add logging to your API:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    logger.info(f"Received prediction request")
    # ... prediction code ...
    logger.info(f"Prediction completed: {predictions[0][0]}")
```

### Metrics

Track important metrics:
- Request count
- Response time
- Prediction confidence
- Error rate
- Model accuracy over time

### Health Checks

Implement health check endpoint:
```python
@app.route('/health', methods=['GET'])
def health():
    try:
        # Test model
        test_pred = predictor.predict("test")
        return jsonify({
            'status': 'healthy',
            'model_loaded': True
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500
```

---

## Scaling

### Horizontal Scaling

Use load balancer with multiple instances:
```
Load Balancer
    ├── Instance 1
    ├── Instance 2
    └── Instance 3
```

### Caching

Cache frequent predictions:
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_predict(ticket_text):
    return predictor.predict(ticket_text)
```

### Batch Processing

For high volume:
```python
@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    tickets = request.json.get('tickets', [])
    predictions = predictor.predict_batch(tickets)
    return jsonify({'predictions': predictions})
```

### GPU Optimization

Enable GPU for faster inference:
```python
# In predict.py
device = 0 if torch.cuda.is_available() else -1
```

---

## Security

### API Authentication

Add API key authentication:
```python
from functools import wraps

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key != 'your-secret-key':
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # ... prediction code ...
```

### Rate Limiting

Implement rate limiting:
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/predict', methods=['POST'])
@limiter.limit("100 per hour")
def predict():
    # ... prediction code ...
```

---

## Production Checklist

- [ ] Model trained and tested
- [ ] API endpoints implemented
- [ ] Error handling added
- [ ] Logging configured
- [ ] Health checks implemented
- [ ] Authentication added
- [ ] Rate limiting configured
- [ ] Monitoring set up
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Backup strategy defined
- [ ] Rollback plan prepared

---

## Troubleshooting

### High Memory Usage
- Reduce batch size
- Use model quantization
- Enable swap memory

### Slow Inference
- Enable GPU
- Use smaller model
- Implement caching
- Optimize batch processing

### Model Loading Issues
- Check file paths
- Verify model files exist
- Check permissions
- Validate model format

---

## Support

For deployment issues, refer to:
- DOCUMENTATION.md for technical details
- QUICKSTART.md for setup instructions
- examples.py for usage patterns
