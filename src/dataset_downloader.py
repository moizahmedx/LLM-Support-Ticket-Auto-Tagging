"""
Dataset Downloader Module

This module handles downloading support ticket datasets from HuggingFace.
It automatically downloads, validates, and saves the dataset locally.
"""

import os
import pandas as pd
from datasets import load_dataset
from pathlib import Path


class DatasetDownloader:
    """
    Downloads and saves support ticket datasets from HuggingFace.
    """
    
    def __init__(self, dataset_name="bitext/Bitext-customer-support-llm-chatbot-training-dataset"):
        """
        Initialize the downloader with dataset name.
        
        Args:
            dataset_name (str): HuggingFace dataset identifier
        """
        self.dataset_name = dataset_name
        self.raw_data_dir = Path("data/raw")
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
    def download(self):
        """
        Download the dataset from HuggingFace and save locally.
        
        Returns:
            pd.DataFrame: Downloaded dataset as a pandas DataFrame
        """
        print(f"Downloading dataset: {self.dataset_name}")
        print("This may take a few minutes...")
        
        try:
            # Load dataset from HuggingFace
            dataset = load_dataset(self.dataset_name, split="train")
            
            # Convert to pandas DataFrame
            df = pd.DataFrame(dataset)
            
            print(f"Dataset downloaded successfully!")
            print(f"Total records: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            
            # Save to CSV
            output_path = self.raw_data_dir / "support_tickets.csv"
            df.to_csv(output_path, index=False)
            print(f"Dataset saved to: {output_path}")
            
            # Display sample
            print("\nSample records:")
            print(df.head(3))
            
            # Display category distribution
            if 'category' in df.columns:
                print("\nCategory distribution:")
                print(df['category'].value_counts())
            
            return df
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("\nTrying alternative dataset...")
            return self._download_alternative()
    
    def _download_alternative(self):
        """
        Download an alternative dataset if the primary one fails.
        
        Returns:
            pd.DataFrame: Alternative dataset
        """
        try:
            # Try alternative dataset
            alt_dataset = "banking77"
            print(f"Downloading alternative dataset: {alt_dataset}")
            dataset = load_dataset(alt_dataset, split="train")
            df = pd.DataFrame(dataset)
            
            # Rename columns to match expected format
            if 'text' in df.columns and 'label' in df.columns:
                df = df.rename(columns={'text': 'instruction', 'label': 'category'})
            
            output_path = self.raw_data_dir / "support_tickets.csv"
            df.to_csv(output_path, index=False)
            print(f"Alternative dataset saved to: {output_path}")
            
            return df
            
        except Exception as e:
            print(f"Error downloading alternative dataset: {e}")
            raise


def main():
    """
    Main function to download the dataset.
    """
    downloader = DatasetDownloader()
    df = downloader.download()
    
    print("\n" + "="*50)
    print("Dataset download completed successfully!")
    print("="*50)
    print(f"\nNext step: Run 'python src/data_preprocessing.py' to preprocess the data")


if __name__ == "__main__":
    main()
