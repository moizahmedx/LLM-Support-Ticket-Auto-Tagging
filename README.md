# Support Ticket Auto-Tagging Using LLM

A machine learning system for automatically classifying customer support tickets using Large Language Models. The system predicts the top 3 most probable tags for each ticket.

## Features

- Zero-shot classification (no training required)
- Few-shot learning with examples
- Fine-tuned transformer models
- Interactive web dashboard
- Real-time predictions with confidence scores

## Technology Stack

- Python 3.8+
- HuggingFace Transformers
- PyTorch
- Scikit-learn
- Pandas, NumPy
- Streamlit (dashboard)
- Matplotlib, Plotly (visualization)

## Installation

```bash
git clone https://github.com/moizahmedx/LLM-Support-Ticket-Auto-Tagging.git
cd LLM-Support-Ticket-Auto-Tagging
pip install -r requirements.txt
```

## Quick Start

### Launch Dashboard

```bash
streamlit run app.py
```

Open your browser at http://localhost:8501

### Run Pipeline

```bash
# Download dataset
python src/dataset_downloader.py

# Preprocess data
python src/data_preprocessing.py

# Train model (optional)
python src/fine_tuning.py

# Make predictions
python src/predict.py
```

### Quick Demo

```bash
python quick_demo.py
```

## Project Structure

```
├── app.py                    # Streamlit dashboard
├── config.py                 # Configuration
├── requirements.txt          # Dependencies
├── src/                      # Source code
│   ├── dataset_downloader.py
│   ├── data_preprocessing.py
│   ├── zero_shot_classifier.py
│   ├── few_shot_classifier.py
│   ├── fine_tuning.py
│   ├── evaluation.py
│   └── predict.py
├── data/                     # Datasets
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
└── docs/                     # Documentation
```

## Usage Example

```python
from src.predict import TicketPredictor

predictor = TicketPredictor()
ticket = "My laptop overheats and shuts down automatically."
predictions = predictor.predict(ticket, top_k=3)

for tag, confidence in predictions:
    print(f"{tag}: {confidence:.2f}")
```

## Model Performance

- Dataset: 26,872 support tickets
- Categories: 11 (ACCOUNT, ORDER, REFUND, etc.)
- Fine-tuned model accuracy: 85-95%
- Inference speed: ~0.1 seconds per ticket

## Documentation

See the `docs/` folder for detailed documentation:
- QUICKSTART.md - Quick start guide
- DOCUMENTATION.md - Technical documentation
- DASHBOARD_GUIDE.md - Dashboard usage
- DEPLOYMENT.md - Deployment guide

## License

MIT License - see LICENSE file for details

## Author

Moiz Ahmed
