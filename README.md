# MiniLearn

A minimal machine learning library for educational purposes. This project implements various machine learning algorithms from scratch to help understand the underlying concepts.

## Project Structure

```
MiniLearn/
├── linear_model/       # Linear models implementations
├── neighbors/          # Nearest neighbors algorithms
├── cluster/           # Clustering algorithms
├── neural_network/    # Neural network implementations
├── metrics/           # Evaluation metrics
├── utils/            # Utility functions
├── datasets/         # Example datasets
├── tests/            # Unit tests
├── examples/         # Example scripts
└── notebooks/        # Jupyter notebooks
```

## Installation

```bash
# Clone the repository
git clone [your-repo-url]

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Features

- Linear Models
  - Linear Regression
  - Polynomial Regression
  - Logistic Regression
- K-Nearest Neighbors
- K-Means Clustering
- Neural Networks
  - Two-layer Neural Network
- Utility Functions
  - Data splitting
  - Feature scaling
  - Activation functions

## Usage

Check the `examples/` directory for usage examples of each algorithm.

## Testing

Run tests using pytest:

```bash
pytest tests/
```

## License

[Your chosen license]
