"""
Quick Demo Script

Demonstrates the project functionality without downloading large models.
Uses a lightweight approach for quick testing.
"""

import pandas as pd
from pathlib import Path


def demo_data_pipeline():
    """
    Demonstrate the data pipeline.
    """
    print("="*60)
    print("DEMO: Data Pipeline")
    print("="*60)
    
    # Check if data exists
    train_path = Path("data/processed/train.csv")
    if not train_path.exists():
        print("\n❌ Processed data not found.")
        print("Run: python src/data_preprocessing.py")
        return
    
    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    
    print(f"\n✅ Data loaded successfully!")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Validation: {len(val_df)} samples")
    print(f"   Test: {len(test_df)} samples")
    
    # Show categories
    categories = sorted(train_df['category'].unique())
    print(f"\n📊 Categories ({len(categories)}):")
    for cat in categories:
        count = len(train_df[train_df['category'] == cat])
        print(f"   • {cat}: {count} samples")
    
    # Show sample tickets
    print(f"\n📝 Sample Tickets:")
    for i, row in train_df.head(5).iterrows():
        print(f"\n   {i+1}. Category: {row['category']}")
        print(f"      Text: {row['ticket_text'][:80]}...")


def demo_simple_classifier():
    """
    Demonstrate a simple rule-based classifier (no ML needed).
    """
    print("\n" + "="*60)
    print("DEMO: Simple Rule-Based Classifier")
    print("="*60)
    
    # Simple keyword-based classifier
    category_keywords = {
        'ACCOUNT': ['account', 'login', 'password', 'username', 'sign in'],
        'ORDER': ['order', 'purchase', 'buy', 'bought'],
        'REFUND': ['refund', 'money back', 'return', 'reimburse'],
        'PAYMENT': ['payment', 'pay', 'credit card', 'billing'],
        'SHIPPING': ['shipping', 'delivery', 'ship', 'tracking'],
        'INVOICE': ['invoice', 'receipt', 'bill'],
        'CONTACT': ['contact', 'reach', 'call', 'email'],
        'FEEDBACK': ['feedback', 'review', 'complaint', 'suggestion'],
        'CANCEL': ['cancel', 'cancellation', 'terminate'],
        'SUBSCRIPTION': ['subscription', 'subscribe', 'membership'],
        'DELIVERY': ['delivery', 'deliver', 'arrived']
    }
    
    def simple_predict(text, top_k=3):
        """Simple keyword matching."""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[category] = score
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top K
        if sorted_scores:
            return sorted_scores[:top_k]
        else:
            return [('CONTACT', 1)]  # Default
    
    # Test tickets
    test_tickets = [
        "I can't log into my account",
        "Where is my order?",
        "I want a refund for my purchase",
        "My laptop overheats and shuts down",
        "How do I cancel my subscription?"
    ]
    
    print("\n🎯 Predictions:")
    for ticket in test_tickets:
        predictions = simple_predict(ticket, top_k=3)
        print(f"\n   Ticket: {ticket}")
        print(f"   Top predictions:")
        for i, (category, score) in enumerate(predictions, 1):
            print(f"      {i}. {category} (score: {score})")


def demo_statistics():
    """
    Show dataset statistics.
    """
    print("\n" + "="*60)
    print("DEMO: Dataset Statistics")
    print("="*60)
    
    train_path = Path("data/processed/train.csv")
    if not train_path.exists():
        print("\n❌ Data not found. Run preprocessing first.")
        return
    
    train_df = pd.read_csv(train_path)
    
    # Text length statistics
    train_df['text_length'] = train_df['ticket_text'].str.len()
    
    print(f"\n📊 Text Length Statistics:")
    print(f"   Mean: {train_df['text_length'].mean():.2f} characters")
    print(f"   Median: {train_df['text_length'].median():.2f} characters")
    print(f"   Min: {train_df['text_length'].min()} characters")
    print(f"   Max: {train_df['text_length'].max()} characters")
    
    # Category balance
    print(f"\n📊 Category Distribution:")
    category_counts = train_df['category'].value_counts()
    for category, count in category_counts.items():
        percentage = (count / len(train_df)) * 100
        print(f"   {category:15s}: {count:5d} ({percentage:5.2f}%)")


def main():
    """
    Run all demos.
    """
    print("\n" + "="*60)
    print("SUPPORT TICKET AUTO-TAGGING - QUICK DEMO")
    print("="*60)
    print("\nThis demo shows the project functionality without")
    print("downloading large ML models (which can take time).")
    
    try:
        # Demo 1: Data pipeline
        demo_data_pipeline()
        
        # Demo 2: Simple classifier
        demo_simple_classifier()
        
        # Demo 3: Statistics
        demo_statistics()
        
        print("\n" + "="*60)
        print("DEMO COMPLETE!")
        print("="*60)
        print("\n📚 Next Steps:")
        print("   • For ML models: python src/fine_tuning.py")
        print("   • For examples: python examples.py")
        print("   • For full pipeline: python run_pipeline.py")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you've run:")
        print("   python src/data_preprocessing.py")


if __name__ == "__main__":
    main()
