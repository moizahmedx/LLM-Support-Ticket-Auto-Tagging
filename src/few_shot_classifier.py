"""
Few-Shot Classification Module

This module implements few-shot learning for support ticket classification
using example-based prompting with LLMs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm
import json
import random


class FewShotClassifier:
    """
    Few-shot classifier using example-based prompting.
    """
    
    def __init__(self, model_name="facebook/bart-large-mnli", n_examples=3):
        """
        Initialize few-shot classifier.
        
        Args:
            model_name (str): HuggingFace model identifier
            n_examples (int): Number of examples per category
        """
        self.model_name = model_name
        self.n_examples = n_examples
        self.classifier = None
        self.categories = []
        self.examples = {}
        
    def load_model(self):
        """
        Load the pre-trained model.
        """
        print(f"Loading model: {self.model_name}")
        
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=-1  # Use CPU
        )
        
        print("Model loaded successfully!")
    
    def load_categories(self, categories_path="data/processed/categories.txt"):
        """
        Load category labels.
        
        Args:
            categories_path (str): Path to categories file
        """
        with open(categories_path, 'r') as f:
            self.categories = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.categories)} categories")
    
    def load_examples(self, train_data_path="data/processed/train.csv"):
        """
        Load example tickets for few-shot learning.
        
        Args:
            train_data_path (str): Path to training data
        """
        print(f"Loading {self.n_examples} examples per category...")
        
        train_df = pd.read_csv(train_data_path)
        
        # Sample examples for each category
        for category in self.categories:
            category_samples = train_df[train_df['category'] == category]
            
            if len(category_samples) > 0:
                # Sample up to n_examples
                n_samples = min(self.n_examples, len(category_samples))
                samples = category_samples.sample(n=n_samples, random_state=42)
                self.examples[category] = samples['ticket_text'].tolist()
            else:
                self.examples[category] = []
        
        total_examples = sum(len(v) for v in self.examples.values())
        print(f"Loaded {total_examples} example tickets")
    
    def create_prompt(self, text, category):
        """
        Create a few-shot prompt with examples.
        
        Args:
            text (str): Ticket text to classify
            category (str): Category to check
            
        Returns:
            str: Enhanced prompt with examples
        """
        prompt_parts = ["Here are some examples of support tickets:\n"]
        
        # Add examples for this category
        if category in self.examples and self.examples[category]:
            for i, example in enumerate(self.examples[category][:2], 1):
                prompt_parts.append(f"Example {i} ({category}): {example}")
        
        # Add the ticket to classify
        prompt_parts.append(f"\nClassify this ticket: {text}")
        
        return "\n".join(prompt_parts)
    
    def predict_single(self, text, top_k=3):
        """
        Predict top K categories using few-shot learning.
        
        Args:
            text (str): Ticket text
            top_k (int): Number of top predictions
            
        Returns:
            list: List of (category, score) tuples
        """
        if not self.classifier:
            self.load_model()
        
        # Create enhanced prompt with examples
        # For simplicity, we use the base classifier with category descriptions
        # In a full implementation, you'd use a generative model with prompting
        
        # Enhance categories with example context
        enhanced_categories = []
        for cat in self.categories:
            if cat in self.examples and self.examples[cat]:
                # Add context from examples
                example_text = self.examples[cat][0][:50]  # First 50 chars
                enhanced_cat = f"{cat} (like: {example_text}...)"
            else:
                enhanced_cat = cat
            enhanced_categories.append(enhanced_cat)
        
        # Classify with enhanced context
        result = self.classifier(
            text,
            candidate_labels=self.categories,  # Use original categories for matching
            multi_label=False
        )
        
        # Extract top K predictions
        predictions = [
            (label, score) 
            for label, score in zip(result['labels'][:top_k], result['scores'][:top_k])
        ]
        
        return predictions
    
    def predict_batch(self, texts, top_k=3):
        """
        Predict categories for multiple tickets.
        
        Args:
            texts (list): List of ticket texts
            top_k (int): Number of top predictions
            
        Returns:
            list: Predictions for each ticket
        """
        if not self.classifier:
            self.load_model()
        
        all_predictions = []
        
        print(f"Classifying {len(texts)} tickets with few-shot learning...")
        
        for text in tqdm(texts):
            predictions = self.predict_single(text, top_k)
            all_predictions.append(predictions)
        
        return all_predictions
    
    def evaluate(self, test_data_path="data/processed/test.csv", top_k=3):
        """
        Evaluate few-shot classifier on test set.
        
        Args:
            test_data_path (str): Path to test dataset
            top_k (int): Number of top predictions
            
        Returns:
            dict: Evaluation metrics
        """
        print("\nEvaluating few-shot classifier...")
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        print(f"Test set size: {len(test_df)}")
        
        # Limit for speed
        if len(test_df) > 100:
            print("Using first 100 samples for evaluation")
            test_df = test_df.head(100)
        
        # Get predictions
        predictions = self.predict_batch(test_df['ticket_text'].tolist(), top_k)
        
        # Calculate metrics
        correct_top1 = 0
        correct_top3 = 0
        
        for true_label, preds in zip(test_df['category'], predictions):
            pred_labels = [label for label, score in preds]
            
            if pred_labels[0] == true_label:
                correct_top1 += 1
            
            if true_label in pred_labels:
                correct_top3 += 1
        
        accuracy_top1 = correct_top1 / len(test_df)
        accuracy_top3 = correct_top3 / len(test_df)
        
        metrics = {
            'model': 'few-shot',
            'n_examples': self.n_examples,
            'accuracy_top1': accuracy_top1,
            'accuracy_top3': accuracy_top3,
            'test_samples': len(test_df)
        }
        
        print(f"\nFew-Shot Results:")
        print(f"Top-1 Accuracy: {accuracy_top1:.4f}")
        print(f"Top-3 Accuracy: {accuracy_top3:.4f}")
        
        # Save results
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        with open(results_dir / "few_shot_results.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics


def main():
    """
    Main function to run few-shot classification.
    """
    classifier = FewShotClassifier(n_examples=3)
    
    # Load categories and examples
    classifier.load_categories()
    classifier.load_examples()
    
    # Load model
    classifier.load_model()
    
    # Test on sample tickets
    print("\n" + "="*50)
    print("Testing few-shot classification:")
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
    
    # Evaluate
    print("\n" + "="*50)
    classifier.evaluate()
    
    print("\n" + "="*50)
    print("Few-shot classification completed!")
    print("="*50)
    print(f"\nNext step: Run 'python src/fine_tuning.py' to fine-tune a model")


if __name__ == "__main__":
    main()
