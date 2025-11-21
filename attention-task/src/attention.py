"""
Scaled Dot-Product Attention Implementation
Based on "Attention Is All You Need" (Vaswani et al., 2017)

This module implements the scaled dot-product attention mechanism using only NumPy,
which is the core building block of Transformer architectures.
"""

import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute Scaled Dot-Product Attention.
    
    The attention mechanism is computed as:
        Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
    
    Where:
        - Q: Query matrix
        - K: Key matrix
        - V: Value matrix
        - d_k: Dimension of the key vectors (used for scaling)
    
    Args:
        Q (np.ndarray): Query matrix of shape (..., seq_len_q, d_k)
        K (np.ndarray): Key matrix of shape (..., seq_len_k, d_k)
        V (np.ndarray): Value matrix of shape (..., seq_len_v, d_v)
                        Note: seq_len_k must equal seq_len_v
        mask (np.ndarray, optional): Mask matrix to prevent attention to certain positions.
                                     Shape should be broadcastable to (..., seq_len_q, seq_len_k).
                                     Values should be 0 (attend) or 1 (mask out).
                                     Can also use -inf for masked positions.
    
    Returns:
        tuple: (output, attention_weights)
            - output (np.ndarray): Attended output of shape (..., seq_len_q, d_v)
            - attention_weights (np.ndarray): Attention weights after softmax,
                                              shape (..., seq_len_q, seq_len_k)
    
    Raises:
        ValueError: If input dimensions are incompatible
    """
    # Step 1: Validate input dimensions
    if K.shape[-2] != V.shape[-2]:
        raise ValueError(
            f"Key and Value must have the same sequence length. "
            f"Got K: {K.shape[-2]}, V: {V.shape[-2]}"
        )
    
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError(
            f"Query and Key must have the same depth (d_k). "
            f"Got Q: {Q.shape[-1]}, K: {K.shape[-1]}"
        )
    
    # Step 2: Extract the depth of the key vectors (d_k) for scaling
    # This is the last dimension of Q (or K)
    d_k = Q.shape[-1]
    
    # Step 3: Compute attention scores by taking the dot product of Q and K^T
    # Q shape: (..., seq_len_q, d_k)
    # K^T shape: (..., d_k, seq_len_k)
    # scores shape: (..., seq_len_q, seq_len_k)
    scores = np.matmul(Q, K.transpose(*range(K.ndim - 2), -1, -2))
    
    # Step 4: Scale the scores by 1/sqrt(d_k)
    # This prevents the dot products from growing too large, which would push
    # the softmax function into regions with extremely small gradients
    scores = scores / np.sqrt(d_k)
    
    # Step 5: Apply the mask (if provided)
    # Masked positions are set to a very large negative number (-1e9)
    # so that after softmax, their weights become approximately zero
    if mask is not None:
        # Convert boolean mask or 0/1 mask to large negative values
        # We use np.where to replace masked positions (where mask is True or 1)
        # with -1e9 (a very large negative number)
        if mask.dtype == bool:
            scores = np.where(mask, -1e9, scores)
        else:
            # For numeric masks, assume 1 means mask, 0 means attend
            scores = np.where(mask == 1, -1e9, scores)
    
    # Step 6: Apply softmax to get attention weights
    # Softmax is computed along the last axis (seq_len_k dimension)
    # This ensures that for each query, the attention weights sum to 1
    attention_weights = softmax(scores, axis=-1)
    
    # Step 7: Compute the weighted sum of values using attention weights
    # attention_weights shape: (..., seq_len_q, seq_len_k)
    # V shape: (..., seq_len_k, d_v)
    # output shape: (..., seq_len_q, d_v)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


def softmax(x, axis=-1):
    """
    Compute the softmax function in a numerically stable way.
    
    The softmax function converts a vector of real numbers into a probability
    distribution. The standard formula is:
        softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
    
    To avoid numerical overflow with large values, we use the trick:
        softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    This is mathematically equivalent but numerically stable.
    
    Args:
        x (np.ndarray): Input array
        axis (int): Axis along which to compute softmax
    
    Returns:
        np.ndarray: Softmax probabilities with the same shape as input
    """
    # Step 1: Subtract the maximum value for numerical stability
    # This prevents overflow when computing exp() of large numbers
    x_max = np.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    
    # Step 2: Compute the exponential of the shifted values
    exp_x = np.exp(x_shifted)
    
    # Step 3: Compute the sum of exponentials along the specified axis
    sum_exp_x = np.sum(exp_x, axis=axis, keepdims=True)
    
    # Step 4: Divide each exponential by the sum to get probabilities
    # Each row (along the specified axis) will sum to 1
    return exp_x / sum_exp_x


# Additional utility functions for visualization and analysis

def visualize_attention(attention_weights, query_labels=None, key_labels=None):
    """
    Print a text-based visualization of attention weights.
    
    Args:
        attention_weights (np.ndarray): 2D attention weights matrix
        query_labels (list, optional): Labels for query positions
        key_labels (list, optional): Labels for key positions
    """
    if attention_weights.ndim != 2:
        print("Warning: Can only visualize 2D attention weights")
        return
    
    seq_len_q, seq_len_k = attention_weights.shape
    
    # Default labels if not provided
    if query_labels is None:
        query_labels = [f"Q{i}" for i in range(seq_len_q)]
    if key_labels is None:
        key_labels = [f"K{i}" for i in range(seq_len_k)]
    
    # Print header
    print("\nAttention Weights Visualization:")
    print("-" * (10 + seq_len_k * 8))
    print(f"{'':10}", end="")
    for label in key_labels:
        print(f"{label:>8}", end="")
    print()
    print("-" * (10 + seq_len_k * 8))
    
    # Print each row
    for i, (label, row) in enumerate(zip(query_labels, attention_weights)):
        print(f"{label:10}", end="")
        for weight in row:
            print(f"{weight:8.4f}", end="")
        print()
    print("-" * (10 + seq_len_k * 8))


def attention_summary(attention_weights):
    """
    Provide statistical summary of attention weights.
    
    Args:
        attention_weights (np.ndarray): Attention weights matrix
    
    Returns:
        dict: Dictionary containing summary statistics
    """
    return {
        'shape': attention_weights.shape,
        'min': np.min(attention_weights),
        'max': np.max(attention_weights),
        'mean': np.mean(attention_weights),
        'std': np.std(attention_weights),
        'row_sums': np.sum(attention_weights, axis=-1),  # Should all be ~1.0
    }
