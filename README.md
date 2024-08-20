# LSNMF

This package implements Zero-inflated Supervised Non-negative Matrix Factorization (ZISNMF) for Python using PyTorch. ZISNMF is a supervised matrix factorization method that can discover marker genes associated with specific contexts such as cell types and disease states from single-cell RNA-seq data.

## Installation

Users can download the code and install it using pip:
```bash
pip install .
```

## Usage

The usage of `zisnmf` is similar to scikit-learn's `NMF`:

```python
from zisnmf import ZISNMF

# Load your data matrix X and label matrix L
# ...

# Create a LSNMF instance
model = ZISNMF(n_cells, n_features, n_classes, n_extra_states,  zero_inflated=True, device='cuda')

# Fit the model
num_epochs = 30
batch_size = 2048
learning_rate = 0.001
alpha = 0.2
W, H = model.fit_transform(X, L, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate, alpha=alpha)