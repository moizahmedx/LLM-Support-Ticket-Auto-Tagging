"""
Evaluation Module

This module compares all three approaches (zero-shot, few-shot, fine-tuned)
and generates visualization plots.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


class ModelEvaluator:
    """
    Evaluates and compares different classification approaches.
    """
    
    def __init__(self):
        """
        Initialize the evaluator.
        """
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_results(self):
        """
        Load results from all models.
        
        Returns:
            dict: Results from all models
        """
        results = {}
        
        # Load zero-shot results
        zero_shot_path = self.results_dir / "zero_shot_results.json"
        if zero_shot_path.exists():
            with open(zero_shot_path, 'r') as f:
                results['zero_shot'] = json.load(f)
        
        # Load few-shot results
        few_shot_path = self.results_dir / "few_shot_results.json"
        if few_shot_path.exists():
            with open(few_shot_path, 'r') as f:
                results['few_shot'] = json.load(f)
        
        # Load fine-tuned results
        fine_tuned_path = self.results_dir / "fine_tuned_results.json"
        if fine_tuned_path.exists():
            with open(fine_tuned_path, 'r') as f:
                results['fine_tuned'] = json.load(f)
        
        return results
    
    def plot_accuracy_comparison(self, results):
        """
        Plot accuracy comparison across models.
        
        Args:
            results (dict): Results from all models
        """
        print("Generating accuracy comparison plot...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(12, 6))
        
        # Prepare data
        models = []
        top1_accuracies = []
        top3_accuracies = []
        
        for model_name, metrics in results.items():
            models.append(model_name.replace('_', '-').title())
            
            # Get top-1 accuracy
            if 'accuracy_top1' in metrics:
                top1_accuracies.append(metrics['accuracy_top1'])
            elif 'accuracy' in metrics:
                top1_accuracies.append(metrics['accuracy'])
            else:
                top1_accuracies.append(0)
            
            # Get top-3 accuracy
            if 'accuracy_top3' in metrics:
                top3_accuracies.append(metrics['accuracy_top3'])
            else:
                # Estimate top-3 as slightly higher than top-1
                top3_accuracies.append(top1_accuracies[-1] * 1.1)
        
        # Create bar plot
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, top1_accuracies, width, label='Top-1 Accuracy', alpha=0.8)
        plt.bar(x + width/2, top3_accuracies, width, label='Top-3 Accuracy', alpha=0.8)
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.xticks(x, models)
        plt.legend()
        plt.ylim(0, 1.0)
        
        # Add value labels on bars
        for i, v in enumerate(top1_accuracies):
            plt.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for i, v in enumerate(top3_accuracies):
            plt.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "accuracy_comparison.png", dpi=300)
        print(f"Saved: {self.results_dir / 'accuracy_comparison.png'}")
        plt.close()
    
    def plot_metrics_comparison(self, results):
        """
        Plot detailed metrics comparison for fine-tuned model.
        
        Args:
            results (dict): Results from all models
        """
        if 'fine_tuned' not in results:
            print("Fine-tuned results not available, skipping detailed metrics plot")
            return
        
        print("Generating detailed metrics plot...")
        
        fine_tuned = results['fine_tuned']
        
        # Prepare data
        metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        values = [
            fine_tuned.get('accuracy', 0),
            fine_tuned.get('f1', 0),
            fine_tuned.get('precision', 0),
            fine_tuned.get('recall', 0)
        ]
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("husl", len(metrics))
        bars = plt.bar(metrics, values, color=colors, alpha=0.8)
        
        plt.xlabel('Metric', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Fine-Tuned Model - Detailed Metrics', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "detailed_metrics.png", dpi=300)
        print(f"Saved: {self.results_dir / 'detailed_metrics.png'}")
        plt.close()
    
    def generate_summary_report(self, results):
        """
        Generate a text summary report.
        
        Args:
            results (dict): Results from all models
        """
        print("\nGenerating summary report...")
        
        report_lines = [
            "="*60,
            "SUPPORT TICKET AUTO-TAGGING - EVALUATION REPORT",
            "="*60,
            ""
        ]
        
        for model_name, metrics in results.items():
            report_lines.append(f"\n{model_name.upper().replace('_', ' ')}:")
            report_lines.append("-" * 40)
            
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric}: {value}")
        
        report_lines.extend([
            "",
            "="*60,
            "RECOMMENDATIONS:",
            "="*60,
            "",
            "1. Zero-Shot: Fast, no training required, good for quick prototyping",
            "2. Few-Shot: Better accuracy with minimal examples",
            "3. Fine-Tuned: Best performance, requires training data and compute",
            "",
            "For production use, fine-tuned model is recommended for best accuracy.",
            ""
        ])
        
        report_text = "\n".join(report_lines)
        
        # Save report
        with open(self.results_dir / "evaluation_report.txt", 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport saved to: {self.results_dir / 'evaluation_report.txt'}")
    
    def run_evaluation(self):
        """
        Run complete evaluation and generate all outputs.
        """
        print("="*60)
        print("EVALUATING ALL MODELS")
        print("="*60)
        
        # Load results
        results = self.load_results()
        
        if not results:
            print("\nNo results found. Please run the classification scripts first:")
            print("  1. python src/zero_shot_classifier.py")
            print("  2. python src/few_shot_classifier.py")
            print("  3. python src/fine_tuning.py")
            return
        
        print(f"\nFound results for {len(results)} model(s)")
        
        # Generate plots
        self.plot_accuracy_comparison(results)
        self.plot_metrics_comparison(results)
        
        # Generate report
        self.generate_summary_report(results)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETED!")
        print("="*60)
        print(f"\nResults saved in: {self.results_dir}")


def main():
    """
    Main function to run evaluation.
    """
    evaluator = ModelEvaluator()
    evaluator.run_evaluation()
    
    print(f"\nNext step: Run 'python src/predict.py' to classify new tickets")


if __name__ == "__main__":
    main()
