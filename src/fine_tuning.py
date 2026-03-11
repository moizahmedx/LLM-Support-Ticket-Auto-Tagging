"""
Fine-Tuning Module

This module handles fine-tuning a transformer model for support ticket classification.
Uses DistilBERT for efficient training and inference.
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json


class TicketClassifierTrainer:
    """
    Fine-tunes a transformer model for ticket classification.
    """
    
    def __init__(self, model_name="distilbert-base-uncased", max_length=128):
        """
        Initialize the trainer.
        
        Args:
            model_name (str): Pre-trained model to fine-tune
            max_length (int): Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.label2id = {}
        self.id2label = {}
        self.num_labels = 0
        
    def load_data(self):
        """
        Load and prepare training data.
        
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        print("Loading datasets...")
        
        train_df = pd.read_csv("data/processed/train.csv")
        val_df = pd.read_csv("data/processed/val.csv")
        test_df = pd.read_csv("data/processed/test.csv")
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        # Create label mappings
        unique_labels = sorted(train_df['category'].unique())
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.num_labels = len(unique_labels)
        
        print(f"Number of categories: {self.num_labels}")
        
        # Convert to HuggingFace datasets
        train_dataset = self._prepare_dataset(train_df)
        val_dataset = self._prepare_dataset(val_df)
        test_dataset = self._prepare_dataset(test_df)
        
        return train_dataset, val_dataset, test_dataset
    
    def _prepare_dataset(self, df):
        """
        Convert DataFrame to HuggingFace Dataset with tokenization.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dataset: Tokenized dataset
        """
        # Convert labels to IDs
        df['label'] = df['category'].map(self.label2id)
        
        # Create dataset
        dataset = Dataset.from_pandas(df[['ticket_text', 'label']])
        
        # Tokenize
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['ticket_text'],
                padding=False,
                truncation=True,
                max_length=self.max_length
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def initialize_model(self):
        """
        Initialize the model for training.
        """
        print(f"Initializing model: {self.model_name}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        print("Model initialized successfully!")
    
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Predictions from the model
            
        Returns:
            dict: Computed metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted'),
            'precision': precision_score(labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(labels, predictions, average='weighted', zero_division=0)
        }
    
    def train(self, train_dataset, val_dataset, output_dir="models/fine_tuned"):
        """
        Fine-tune the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir (str): Directory to save the model
        """
        print("\nStarting fine-tuning...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            save_total_limit=2
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        print("Training in progress...")
        trainer.train()
        
        # Save final model
        print(f"\nSaving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(f"{output_dir}/label_mappings.json", 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label
            }, f, indent=2)
        
        print("Training completed!")
        
        return trainer
    
    def evaluate(self, trainer, test_dataset):
        """
        Evaluate the fine-tuned model.
        
        Args:
            trainer: Trained model
            test_dataset: Test dataset
            
        Returns:
            dict: Evaluation metrics
        """
        print("\nEvaluating on test set...")
        
        results = trainer.evaluate(test_dataset)
        
        print("\nTest Results:")
        print(f"Accuracy: {results['eval_accuracy']:.4f}")
        print(f"F1 Score: {results['eval_f1']:.4f}")
        print(f"Precision: {results['eval_precision']:.4f}")
        print(f"Recall: {results['eval_recall']:.4f}")
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        metrics = {
            'model': 'fine-tuned',
            'base_model': self.model_name,
            'accuracy': results['eval_accuracy'],
            'f1': results['eval_f1'],
            'precision': results['eval_precision'],
            'recall': results['eval_recall']
        }
        
        with open(results_dir / "fine_tuned_results.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics


def main():
    """
    Main function to fine-tune the model.
    """
    print("="*50)
    print("Fine-Tuning Support Ticket Classifier")
    print("="*50)
    
    # Initialize trainer
    trainer_obj = TicketClassifierTrainer()
    
    # Load data
    train_dataset, val_dataset, test_dataset = trainer_obj.load_data()
    
    # Initialize model
    trainer_obj.initialize_model()
    
    # Train
    trainer = trainer_obj.train(train_dataset, val_dataset)
    
    # Evaluate
    trainer_obj.evaluate(trainer, test_dataset)
    
    print("\n" + "="*50)
    print("Fine-tuning completed successfully!")
    print("="*50)
    print(f"\nModel saved to: models/fine_tuned")
    print(f"\nNext step: Run 'python src/evaluation.py' to compare all models")


if __name__ == "__main__":
    main()
