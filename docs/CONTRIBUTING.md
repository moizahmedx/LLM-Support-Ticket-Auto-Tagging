# Contributing Guide

Thank you for your interest in contributing to the Support Ticket Auto-Tagging project!

## How to Contribute

### 1. Code Contributions

#### Adding New Features
- Create a new module in the `src/` directory
- Follow the existing code structure and style
- Add comprehensive docstrings and comments
- Update documentation accordingly

#### Improving Existing Code
- Maintain backward compatibility
- Add tests for your changes
- Update relevant documentation

### 2. Documentation

- Fix typos or unclear explanations
- Add more examples
- Improve code comments
- Create tutorials or guides

### 3. Bug Reports

When reporting bugs, please include:
- Python version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages or logs

### 4. Feature Requests

- Describe the feature clearly
- Explain the use case
- Suggest implementation approach if possible

## Code Style Guidelines

### Python Code Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to all functions and classes
- Keep functions focused and modular
- Maximum line length: 100 characters

### Documentation Style
- Use clear, concise language
- Include code examples
- Explain the "why" not just the "what"
- Keep documentation up to date with code

## Development Setup

1. Fork the repository
2. Clone your fork
3. Install dependencies: `pip install -r requirements.txt`
4. Create a new branch: `git checkout -b feature-name`
5. Make your changes
6. Test your changes
7. Commit with clear messages
8. Push and create a pull request

## Testing

Before submitting:
- Run `python test_installation.py`
- Test your changes manually
- Ensure all existing functionality still works

## Project Structure

```
src/
├── dataset_downloader.py    # Dataset management
├── data_preprocessing.py    # Data cleaning
├── zero_shot_classifier.py  # Zero-shot model
├── few_shot_classifier.py   # Few-shot model
├── fine_tuning.py           # Model training
├── evaluation.py            # Model evaluation
└── predict.py               # Inference interface
```

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Be descriptive but concise
- Reference issues when applicable

Examples:
- `Add batch prediction support`
- `Fix memory leak in fine-tuning`
- `Update documentation for zero-shot classifier`

## Questions?

Feel free to open an issue for any questions or clarifications.

Thank you for contributing!
