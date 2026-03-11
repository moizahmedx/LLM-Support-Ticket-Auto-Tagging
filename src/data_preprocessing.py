"""
Data Preprocessing Module

This module handles cleaning, preprocessing, and splitting the support ticket dataset.
It prepares the data for training and evaluation.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter


class DataPreprocessor:
    """
    Preprocesses support ticket data for ML models.
    """
    
    def __init__(self, raw_data_path="data/raw/support_tickets.csv"):
        """
        Initialize preprocessor with data path.
        
        Args:
            raw_data_path (str): Path to raw dataset
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_dir = Path("data/processed")
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """
        Load raw dataset from CSV.
        
        Returns:
            pd.DataFrame: Raw dataset
        """
        print(f"Loading data from: {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        print(f"Loaded {len(df)} records")
        return df
    
    def clean_text(self, text):
        """
        Clean and normalize text data.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?-]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
    
    def preprocess(self, df):
        """
        Apply preprocessing steps to the dataset.
        
        Args:
            df (pd.DataFrame): Raw dataset
            
        Returns:
            pd.DataFrame: Preprocessed dataset
        """
        print("\nPreprocessing data...")
        
        # Identify text and label columns
        text_col = self._identify_text_column(df)
        label_col = self._identify_label_column(df)
        
        print(f"Text column: {text_col}")
        print(f"Label column: {label_col}")
        
        # Create standardized dataframe
        processed_df = pd.DataFrame()
        processed_df['ticket_id'] = range(len(df))
        processed_df['ticket_text'] = df[text_col].apply(self.clean_text)
        processed_df['category'] = df[label_col]
        
        # Remove empty texts
        processed_df = processed_df[processed_df['ticket_text'].str.len() > 0]
        
        # Remove duplicates
        initial_count = len(processed_df)
        processed_df = processed_df.drop_duplicates(subset=['ticket_text'])
        print(f"Removed {initial_count - len(processed_df)} duplicate records")
        
        # Handle missing values
        processed_df = processed_df.dropna()
        
        print(f"Final dataset size: {len(processed_df)} records")
        
        return processed_df
    
    def _identify_text_column(self, df):
        """
        Identify the column containing ticket text.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            str: Column name
        """
        possible_names = ['instruction', 'text', 'ticket_text', 'description', 'query']
        for col in possible_names:
            if col in df.columns:
                return col
        # Return first string column
        return df.select_dtypes(include=['object']).columns[0]
    
    def _identify_label_column(self, df):
        """
        Identify the column containing categories/labels.
        
        Args:
            df (pd.DataFrame): Dataset
            
        Returns:
            str: Column name
        """
        possible_names = ['category', 'label', 'intent', 'class', 'tag']
        for col in possible_names:
            if col in df.columns:
                return col
        # Return second string column or first numeric
        string_cols = df.select_dtypes(include=['object']).columns
        if len(string_cols) > 1:
            return string_cols[1]
        return df.select_dtypes(include=['int', 'float']).columns[0]
    
    def split_data(self, df, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            df (pd.DataFrame): Preprocessed dataset
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set
            random_state (int): Random seed
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        print("\nSplitting data...")
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['category'] if df['category'].nunique() < len(df) * 0.5 else None
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val['category'] if train_val['category'].nunique() < len(train_val) * 0.5 else None
        )
        
        print(f"Train set: {len(train)} records")
        print(f"Validation set: {len(val)} records")
        print(f"Test set: {len(test)} records")
        
        return train, val, test
    
    def save_processed_data(self, train, val, test):
        """
        Save processed datasets to CSV files.
        
        Args:
            train (pd.DataFrame): Training set
            val (pd.DataFrame): Validation set
            test (pd.DataFrame): Test set
        """
        print("\nSaving processed data...")
        
        train.to_csv(self.processed_data_dir / "train.csv", index=False)
        val.to_csv(self.processed_data_dir / "val.csv", index=False)
        test.to_csv(self.processed_data_dir / "test.csv", index=False)
        
        print(f"Data saved to: {self.processed_data_dir}")
        
        # Save category mapping
        categories = sorted(train['category'].unique())
        with open(self.processed_data_dir / "categories.txt", 'w') as f:
            for cat in categories:
                f.write(f"{cat}\n")
        
        print(f"Total categories: {len(categories)}")
        print("\nCategory distribution in training set:")
        print(train['category'].value_counts().head(10))


def main():
    """
    Main function to preprocess the dataset.
    """
    preprocessor = DataPreprocessor()
    
    # Load raw data
    df = preprocessor.load_data()
    
    # Preprocess
    processed_df = preprocessor.preprocess(df)
    
    # Split data
    train, val, test = preprocessor.split_data(processed_df)
    
    # Save processed data
    preprocessor.save_processed_data(train, val, test)
    
    print("\n" + "="*50)
    print("Data preprocessing completed successfully!")
    print("="*50)
    print(f"\nNext step: Run 'python src/zero_shot_classifier.py' to test zero-shot classification")


if __name__ == "__main__":
    main()
