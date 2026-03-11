"""
Complete Pipeline Runner

This script runs the entire ML pipeline from start to finish.
Use this for automated execution of all steps.
"""

import sys
import subprocess
from pathlib import Path


def run_command(command, description):
    """
    Run a command and handle errors.
    
    Args:
        command (list): Command to run
        description (str): Description of the step
    """
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✓ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed!")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error in {description}")
        print(f"Error: {e}")
        return False


def main():
    """
    Run the complete pipeline.
    """
    print("="*60)
    print("SUPPORT TICKET AUTO-TAGGING - COMPLETE PIPELINE")
    print("="*60)
    print("\nThis will run all steps from data download to evaluation.")
    print("Estimated time: 30-60 minutes depending on your hardware.")
    print("\nPress Ctrl+C to cancel at any time.")
    
    input("\nPress Enter to continue...")
    
    steps = [
        {
            'command': [sys.executable, 'src/dataset_downloader.py'],
            'description': 'Download Dataset',
            'required': True
        },
        {
            'command': [sys.executable, 'src/data_preprocessing.py'],
            'description': 'Preprocess Data',
            'required': True
        },
        {
            'command': [sys.executable, 'src/zero_shot_classifier.py'],
            'description': 'Zero-Shot Classification',
            'required': False
        },
        {
            'command': [sys.executable, 'src/few_shot_classifier.py'],
            'description': 'Few-Shot Classification',
            'required': False
        },
        {
            'command': [sys.executable, 'src/fine_tuning.py'],
            'description': 'Fine-Tune Model',
            'required': True
        },
        {
            'command': [sys.executable, 'src/evaluation.py'],
            'description': 'Evaluate Models',
            'required': False
        }
    ]
    
    completed = []
    failed = []
    
    for step in steps:
        success = run_command(step['command'], step['description'])
        
        if success:
            completed.append(step['description'])
        else:
            failed.append(step['description'])
            
            if step['required']:
                print("\n" + "="*60)
                print("PIPELINE STOPPED - Required step failed")
                print("="*60)
                print(f"\nFailed at: {step['description']}")
                print("\nPlease fix the error and run again.")
                return 1
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    
    print(f"\nCompleted steps: {len(completed)}")
    for step in completed:
        print(f"  ✓ {step}")
    
    if failed:
        print(f"\nFailed steps: {len(failed)}")
        for step in failed:
            print(f"  ✗ {step}")
    
    print("\n" + "="*60)
    
    if not failed:
        print("SUCCESS! Pipeline completed successfully!")
        print("="*60)
        print("\nYou can now:")
        print("  1. Check results in the 'results/' directory")
        print("  2. Run predictions: python src/predict.py")
        print("  3. Explore the Jupyter notebook: jupyter notebook notebooks/experiments.ipynb")
        return 0
    else:
        print("PARTIAL SUCCESS - Some optional steps failed")
        print("="*60)
        print("\nCore functionality should still work.")
        print("You can run failed steps individually if needed.")
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nPipeline cancelled by user.")
        sys.exit(1)
