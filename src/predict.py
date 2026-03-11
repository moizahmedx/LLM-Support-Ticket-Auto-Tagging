"""
Prediction Module

This module provides a simple interface for predicting tags on new support tickets.
Uses the fine-tuned model by default, with fallback to zero-shot.
"""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import warnings
warnings.filterwarnings('ignore')


class TicketPredictor:
    """
    Predicts tags for support tickets using trained models.
    """
    
    def __init__(self, model_path="models/fine_tuned", use_zero_shot=False):
        """
        Initialize the predictor.
        
        Args:
            model_path (str): Path to fine-tuned model
            use_zero_shot (bool): Use zero-shot instead of fine-tuned
        """
        self.model_path = Path(model_path)
        self.use_zero_shot = use_zero_shot
        self.model = None
        self.tokenizer = None
        self.id2label = {}
        self.categories = []
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """
        Load the classification model.
        """
        if self.use_zero_shot or not self.model_path.exists():
            print("Loading zero-shot model...")
            self.model = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1
            )
            self._load_categories()
            print("Zero-shot model loaded!")
        else:
            print(f"Loading fine-tuned model from {self.model_path}...")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            
            # Load label mappings
            label_path = self.model_path / "label_mappings.json"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    mappings = json.load(f)
                    self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
            
            print("Fine-tuned model loaded!")
    
    def _load_categories(self):
        """
        Load category list for zero-shot classification.
        """
        categories_path = Path("data/processed/categories.txt")
        if categories_path.exists():
            with open(categories_path, 'r') as f:
                self.categories = [line.strip() for line in f.readlines()]
        else:
            # Default categories
            self.categories = [
                'technical_support', 'billing', 'account', 'network',
                'hardware', 'software', 'password_reset', 'installation'
            ]
    
    def predict(self, ticket_text, top_k=3):
        """
        Predict top K tags for a support ticket.
        
        Args:
            ticket_text (str): Support ticket text
            top_k (int): Number of top predictions to return
            
        Returns:
            list: List of (tag, confidence) tuples
        """
        if self.use_zero_shot or isinstance(self.model, pipeline):
            return self._predict_zero_shot(ticket_text, top_k)
        else:
            return self._predict_fine_tuned(ticket_text, top_k)
    
    def _predict_zero_shot(self, text, top_k):
        """
        Predict using zero-shot classification.
        
        Args:
            text (str): Ticket text
            top_k (int): Number of predictions
            
        Returns:
            list: Predictions
        """
        result = self.model(text, candidate_labels=self.categories, multi_label=False)
        
        predictions = [
            (label, score)
            for label, score in zip(result['labels'][:top_k], result['scores'][:top_k])
        ]
        
        return predictions
    
    def _predict_fine_tuned(self, text, top_k):
        """
        Predict using fine-tuned model.
        
        Args:
            text (str): Ticket text
            top_k (int): Number of predictions
            
        Returns:
            list: Predictions
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
        
        # Get top K predictions
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))
        
        predictions = [
            (self.id2label[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return predictions
    
    def predict_batch(self, ticket_texts, top_k=3):
        """
        Predict tags for multiple tickets.
        
        Args:
            ticket_texts (list): List of ticket texts
            top_k (int): Number of predictions per ticket
            
        Returns:
            list: Predictions for each ticket
        """
        return [self.predict(text, top_k) for text in ticket_texts]


def main():
    """
    Main function for interactive prediction.
    """
    print("="*60)
    print("SUPPORT TICKET AUTO-TAGGING - PREDICTION")
    print("="*60)
    
    # Initialize predictor
    try:
        predictor = TicketPredictor()
    except Exception as e:
        print(f"\nCould not load fine-tuned model: {e}")
        print("Falling back to zero-shot classification...")
        predictor = TicketPredictor(use_zero_shot=True)
    
    # Example tickets
    sample_tickets = [
        "My internet connection keeps disconnecting every few minutes.",
        "I forgot my password and the reset link is not working.",
        "The application crashes when I try to open large files.",
        "My laptop overheats and shuts down automatically.",
        "I need help setting up my email account on the new device.",
        "The printer is not responding when I try to print documents."
    ]
    
    print("\nPredicting tags for sample tickets:\n")
    
    for i, ticket in enumerate(sample_tickets, 1):
        print(f"{i}. Ticket: {ticket}")
        predictions = predictor.predict(ticket, top_k=3)
        
        print("   Top 3 Predicted Tags:")
        for rank, (tag, confidence) in enumerate(predictions, 1):
            print(f"      {rank}. {tag} (confidence: {confidence:.4f})")
        print()
    
    # Interactive mode
    print("="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter a support ticket (or 'quit' to exit):\n")
    
    while True:
        ticket = input("Ticket: ").strip()
        
        if ticket.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not ticket:
            continue
        
        predictions = predictor.predict(ticket, top_k=3)
        
        print("\nTop 3 Predicted Tags:")
        for rank, (tag, confidence) in enumerate(predictions, 1):
            print(f"  {rank}. {tag} (confidence: {confidence:.4f})")
        print()


if __name__ == "__main__":
    main()
