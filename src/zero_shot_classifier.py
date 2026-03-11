"""
Zero-Shot Classification Module

This module implements zero-shot classification for support tickets using
pre-trained models without any fine-tuning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm
import json


class ZeroShotClassifier:
    """
    Zero-shot classifier for support tickets using pre-trained models.
    """
    
    def __init__(self, model_name="facebook/bart-large-mnli"):
        """
        Initialize zero-shot classifier.
        
        Args:
            model_name (str): HuggingFace model identifier
        """
        self.model_name = model_name
        self.classifier = None
        self.categories = []
        
    def load_model(self):
        """
        Load the pre-trained zero-shot classification model.
        """
        print(f"Loading zero-shot model: {self.model_name}")
        print("This may take a few minutes on first run...")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=-1  # Use CPU; change to 0 for GPU
        )
        
        print("Model loaded successfully!")
    
    def load_categories(self, categories_path="data/processed/categories.txt"):
        """
        Load category labels from file.
        
        Args:
            categories_path (str): Path to categories file
        """
        with open(categories_path, 'r') as f:
            self.categories = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.categories)} categories")
        
        # If too many categories, use top N most common
        if len(self.categories) > 50:
            print("Warning: Too many categories for efficient zero-shot classification")
            print("Consider using fine-tuning approach for better performance")
    
    def predict_single(self, text, top_k=3):
        """
        Predict top K categories for a single ticket.
        
        Args:
            text (str): Ticket text
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of (category, score) tuples
        """
        if not self.classifier:
            self.load_model()
        
        # Perform zero-shot classification
        result = self.classifier(
            text,
            candidate_labels=self.categories,
            multi_label=False
        )
        
        # Extract top K predictions
        predictions = [
            (label, score) 
            for label, score in zip(result['labels'][:top_k], result['scores'][:top_k])
        ]
        
        return predictions
    
    def predict_batch(self, texts, top_k=3, batch_size=8):
        """
        Predict categories for multiple tickets.
        
        Args:
            texts (list): List of ticket texts
            top_k (int): Number of top predictions per ticket
            batch_size (int): Batch size for processing
            
        Returns:
            list: List of predictions for each ticket
        """
        if not self.classifier:
            self.load_model()
        
        all_predictions = []
        
        print(f"Classifying {len(texts)} tickets...")
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            
            for text in batch:
                predictions = self.predict_single(text, top_k)
                all_predictions.append(predictions)
        
        return all_predictions
    
    def evaluate(self, test_data_path="data/processed/test.csv", top_k=3):
        """
        Evaluate zero-shot classifier on test set.
        
        Args:
            test_data_path (str): Path to test dataset
            top_k (int): Number of top predictions to consider
            
        Returns:
            dict: Evaluation metrics
        """
        print("\nEvaluating zero-shot classifier...")
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        print(f"Test set size: {len(test_df)}")
        
        # Limit evaluation to first 100 samples for speed
        if len(test_df) > 100:
            print("Using first 100 samples for evaluation (zero-shot is slow)")
            test_df = test_df.head(100)
        
        # Get predictions
        predictions = self.predict_batch(test_df['ticket_text'].tolist(), top_k)
        
        # Calculate metrics
        correct_top1 = 0
        correct_top3 = 0
        
        for i, (true_label, preds) in enumerate(zip(test_df['category'], predictions)):
            pred_labels = [label for label, score in preds]
            
            if pred_labels[0] == true_label:
                correct_top1 += 1
            
            if true_label in pred_labels:
                correct_top3 += 1
        
        accuracy_top1 = correct_top1 / len(test_df)
        accuracy_top3 = correct_top3 / len(test_df)
        
        metrics = {
            'model': 'zero-shot',
            'accuracy_top1': accuracy_top1,
            'accuracy_top3': accuracy_top3,
            'test_samples': len(test_df)
        }
        
        print(f"\nZero-Shot Results:")
        print(f"Top-1 Accuracy: {accuracy_top1:.4f}")
        print(f"Top-3 Accuracy: {accuracy_top3:.4f}")
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "zero_shot_results.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics


def main():
    """
    Main function to run zero-shot classification.
    """
    classifier = ZeroShotClassifier()
    
    # Load categories
    classifier.load_categories()
    
    # Load model
    classifier.load_model()
    
    # Test on sample tickets
    print("\n" + "="*50)
    print("Testing on sample tickets:")
    print("="*50)
    
    sample_tickets = [
        "My internet connection keeps disconnecting every few minutes.",
        "I forgot my password and the reset link is not working.",
        "The application crashes when I try to open large files.",
        "My laptop overheats and shuts down automatically."
    ]
    
    for ticket in sample_tickets:
        print(f"\nTicket: {ticket}")
        predictions = classifier.predict_single(ticket, top_k=3)
        print("Top 3 predictions:")
        for i, (label, score) in enumerate(predictions, 1):
            print(f"  {i}. {label} (confidence: {score:.4f})")
    
    # Evaluate on test set
    print("\n" + "="*50)
    classifier.evaluate()
    
    print("\n" + "="*50)
    print("Zero-shot classification completed!")
    print("="*50)
    print(f"\nNext step: Run 'python src/few_shot_classifier.py' to test few-shot learning")


if __name__ == "__main__":
    main()
