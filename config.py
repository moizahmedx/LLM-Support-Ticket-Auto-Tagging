"""
Configuration File

Central configuration for the support ticket auto-tagging project.
Modify these settings to customize the project behavior.
"""

# Dataset Configuration
DATASET_CONFIG = {
    'primary_dataset': 'bitext/Bitext-customer-support-llm-chatbot-training-dataset',
    'fallback_dataset': 'banking77',
    'raw_data_path': 'data/raw/support_tickets.csv',
}

# Data Preprocessing Configuration
PREPROCESSING_CONFIG = {
    'test_size': 0.2,
    'val_size': 0.1,
    'random_state': 42,
    'min_text_length': 10,
    'max_text_length': 512,
}

# Model Configuration
MODEL_CONFIG = {
    'zero_shot_model': 'facebook/bart-large-mnli',
    'fine_tune_model': 'distilbert-base-uncased',
    'max_sequence_length': 128,
}

# Training Configuration
TRAINING_CONFIG = {
    'learning_rate': 2e-5,
    'batch_size': 16,
    'num_epochs': 3,
    'weight_decay': 0.01,
    'warmup_steps': 500,
    'logging_steps': 50,
}

# Few-Shot Configuration
FEW_SHOT_CONFIG = {
    'n_examples_per_category': 3,
    'max_categories': 50,
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    'top_k': 3,
    'test_sample_size': 100,  # For zero-shot/few-shot (slow)
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'default_top_k': 3,
    'confidence_threshold': 0.5,
    'use_gpu': False,  # Set to True if GPU available
}

# Paths Configuration
PATHS = {
    'data_raw': 'data/raw',
    'data_processed': 'data/processed',
    'models': 'models',
    'results': 'results',
    'notebooks': 'notebooks',
}
