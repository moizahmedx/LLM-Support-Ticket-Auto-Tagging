# Quick Start Guide

Get started with the Support Ticket Auto-Tagging system in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Internet connection (for downloading models and datasets)

## Installation

1. Navigate to the project directory:
```bash
cd support-ticket-auto-tagging
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

This will install all required packages including PyTorch, Transformers, and scikit-learn.

## Running the Complete Pipeline

### Step 1: Download Dataset (2-3 minutes)
```bash
python src/dataset_downloader.py
```

This downloads a customer support ticket dataset from HuggingFace and saves it to `data/raw/`.

### Step 2: Preprocess Data (< 1 minute)
```bash
python src/data_preprocessing.py
```

Cleans the data, removes duplicates, and splits into train/val/test sets.

### Step 3: Test Zero-Shot Classification (5-10 minutes)
```bash
python src/zero_shot_classifier.py
```

Tests classification without any training using a pre-trained model.

### Step 4: Test Few-Shot Learning (5-10 minutes)
```bash
python src/few_shot_classifier.py
```

Improves accuracy using example-based prompting.

### Step 5: Fine-Tune Model (10-30 minutes depending on hardware)
```bash
python src/fine_tuning.py
```

Trains a custom classifier for best performance.

### Step 6: Evaluate All Models (< 1 minute)
```bash
python src/evaluation.py
```

Compares all approaches and generates visualization plots.

### Step 7: Predict on New Tickets
```bash
python src/predict.py
```

Interactive mode to classify your own support tickets!

## Quick Test (Skip Training)

If you want to test immediately without training:

```bash
# Download and preprocess data
python src/dataset_downloader.py
python src/data_preprocessing.py

# Use zero-shot classification (no training needed)
python src/zero_shot_classifier.py
```

Then modify `src/predict.py` to use zero-shot mode:
```python
predictor = TicketPredictor(use_zero_shot=True)
```

## Using the Jupyter Notebook

For interactive exploration:

```bash
jupyter notebook notebooks/experiments.ipynb
```

## Example Usage in Code

```python
from src.predict import TicketPredictor

# Initialize predictor
predictor = TicketPredictor()

# Predict tags
ticket = "My laptop overheats and shuts down automatically."
tags = predictor.predict(ticket, top_k=3)

# Display results
for i, (tag, confidence) in enumerate(tags, 1):
    print(f"{i}. {tag} (confidence: {confidence:.4f})")
```

## Expected Output

```
Ticket: "My laptop overheats and shuts down automatically."

Top 3 Predicted Tags:
1. hardware (confidence: 0.8734)
2. technical_support (confidence: 0.7245)
3. system_issue (confidence: 0.4521)
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size in `src/fine_tuning.py` (line 120)
- Use a smaller model like `distilbert-base-uncased`

### Slow Performance
- Zero-shot and few-shot are slower (use fine-tuned model for production)
- Enable GPU by changing `device=-1` to `device=0` in classifier files

### Dataset Download Fails
- Check internet connection
- The script will automatically try an alternative dataset

## Project Structure

```
support-ticket-auto-tagging/
├── data/
│   ├── raw/              # Downloaded datasets
│   └── processed/        # Cleaned and split data
├── src/
│   ├── dataset_downloader.py
│   ├── data_preprocessing.py
│   ├── zero_shot_classifier.py
│   ├── few_shot_classifier.py
│   ├── fine_tuning.py
│   ├── evaluation.py
│   └── predict.py
├── notebooks/
│   └── experiments.ipynb
├── models/               # Saved models
├── results/              # Evaluation results and plots
└── requirements.txt
```

## Next Steps

1. Experiment with different models in `src/fine_tuning.py`
2. Adjust hyperparameters for better performance
3. Add more categories or use your own dataset
4. Deploy the model as a REST API
5. Integrate with your ticketing system

## Support

For issues or questions, refer to the main README.md or check the code comments in each module.

Happy tagging! 🎯
