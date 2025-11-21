# Scaled Dot-Product Attention Implementation

This directory contains a pure NumPy implementation of the Scaled Dot-Product Attention mechanism from "Attention Is All You Need" (Vaswani et al., 2017).

## Overview

The scaled dot-product attention is the core building block of Transformer architectures and is computed as:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

Where:
- **Q** (Query): Matrix representing what we're looking for
- **K** (Key): Matrix representing what's available to attend to
- **V** (Value): Matrix containing the actual information
- **d_k**: Dimension of the key vectors (used for scaling)

## Project Structure

```
attention-task/
├── src/
│   ├── attention.py        # Core attention implementation
│   └── demo.py             # Comprehensive demonstrations
├── tests/
│   └── test_attention.py   # Unit tests (16 test cases)
└── README.md               # This file
```

## Installation

Ensure you have NumPy installed:

```bash
pip install numpy pytest
```

Or install from the project requirements:

```bash
cd /media/raj/T7/ML\ assessment/ml-intern-assessment-main
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import numpy as np
from attention import scaled_dot_product_attention

# Create sample Q, K, V matrices
seq_len = 4
d_k = 8

Q = np.random.randn(seq_len, d_k)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

# Compute attention
output, attention_weights = scaled_dot_product_attention(Q, K, V)

print("Output shape:", output.shape)
print("Attention weights shape:", attention_weights.shape)
```

### With Causal Masking

```python
# Create a causal mask (for autoregressive models)
mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)

output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
```

### Batch Processing

```python
# Works seamlessly with batched inputs
batch_size = 32
seq_len = 10
d_k = 64

Q = np.random.randn(batch_size, seq_len, d_k)
K = np.random.randn(batch_size, seq_len, d_k)
V = np.random.randn(batch_size, seq_len, d_k)

output, attention_weights = scaled_dot_product_attention(Q, K, V)
# Output shape: (32, 10, 64)
```

## Running Tests

Run all 16 unit tests to validate the implementation:

```bash
cd attention-task
python -m pytest tests/test_attention.py -v
```

All tests should pass, covering:
- Softmax correctness and numerical stability
- Attention output shapes and values
- Masking functionality
- Batch and multi-dimensional processing
- Edge cases and error handling

## Running Demonstrations

Run the comprehensive demonstration script to see the attention mechanism in action:

```bash
cd attention-task/src
python demo.py
```

This will show 5 different demonstrations:
1. **Simple Attention**: Basic 3×3 example
2. **Causal Masking**: Preventing attention to future positions
3. **Batch Attention**: Processing multiple sequences simultaneously
4. **Multi-Head Attention**: Simulating multiple attention heads
5. **Sentence-Level Attention**: Interpretable example with words

## Key Features

### 1. **Numerically Stable Softmax**
- Subtracts maximum value before exponential to prevent overflow
- Handles large values gracefully

### 2. **Flexible Masking**
- Supports boolean masks and numeric masks
- Can implement causal (autoregressive) attention
- Can mask out padding tokens

### 3. **Arbitrary Dimensionality**
- Works with 2D (simple), 3D (batched), 4D (multi-head), or higher dimensional inputs
- Automatically broadcasts operations correctly

### 4. **Pure NumPy**
- No dependencies on TensorFlow, PyTorch, or other ML frameworks
- Only uses NumPy for all computations

### 5. **Comprehensive Documentation**
- Detailed docstrings for all functions
- Inline comments explaining each step
- Mathematical formulas in documentation

## Implementation Details

### Scaling Factor

The attention scores are scaled by `1/sqrt(d_k)` to prevent the dot products from growing too large. Without scaling, very large dot products would push the softmax function into regions with extremely small gradients, making training difficult.

### Masking

Masking is implemented by setting masked positions to a large negative number (-1e9) before the softmax. This ensures that after softmax, the attention weights for masked positions are approximately zero.

### Attention Weights

The attention weights form a probability distribution over the keys for each query. Each row of the attention matrix sums to 1.0, representing how much each query "attends" to each key.

## Testing Results

All 16 tests pass successfully:

```
tests/test_attention.py::TestSoftmax::test_softmax_sums_to_one PASSED
tests/test_attention.py::TestSoftmax::test_softmax_positive_values PASSED
tests/test_attention.py::TestSoftmax::test_softmax_numerical_stability PASSED
tests/test_attention.py::TestSoftmax::test_softmax_2d PASSED
tests/test_attention.py::TestScaledDotProductAttention::test_basic_attention_shape PASSED
tests/test_attention.py::TestScaledDotProductAttention::test_attention_weights_sum_to_one PASSED
tests/test_attention.py::TestScaledDotProductAttention::test_attention_weights_positive PASSED
tests/test_attention.py::TestScaledDotProductAttention::test_batched_attention PASSED
tests/test_attention.py::TestScaledDotProductAttention::test_attention_with_mask PASSED
tests/test_attention.py::TestScaledDotProductAttention::test_dimension_mismatch_error PASSED
tests/test_attention.py::TestScaledDotProductAttention::test_kv_length_mismatch_error PASSED
tests/test_attention.py::TestScaledDotProductAttention::test_identity_attention PASSED
tests/test_attention.py::TestScaledDotProductAttention::test_uniform_keys PASSED
tests/test_attention.py::TestScaledDotProductAttention::test_scaling_effect PASSED
tests/test_attention.py::TestMultiDimensionalAttention::test_3d_attention PASSED
tests/test_attention.py::TestMultiDimensionalAttention::test_4d_attention PASSED

16 passed in 0.16s
```

## Mathematical Foundation

The attention mechanism computes a weighted sum of values, where the weights are determined by the compatibility between queries and keys:

1. **Compute Scores**: `scores = Q @ K^T`
2. **Scale**: `scores = scores / sqrt(d_k)`
3. **Apply Mask** (optional): `scores[mask] = -inf`
4. **Softmax**: `attention_weights = softmax(scores)`
5. **Weighted Sum**: `output = attention_weights @ V`

This allows each position in the output to "attend" to different positions in the input, with learned attention patterns.

## References

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). "Attention Is All You Need". *Neural Information Processing Systems (NIPS)*.
- The Illustrated Transformer by Jay Alammar
- PyTorch Documentation on Attention Mechanisms
