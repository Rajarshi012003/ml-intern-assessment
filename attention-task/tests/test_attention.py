"""
Unit tests for the Scaled Dot-Product Attention implementation
"""

import pytest
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from attention import scaled_dot_product_attention, softmax


class TestSoftmax:
    """Tests for the softmax function"""
    
    def test_softmax_sums_to_one(self):
        """Test that softmax outputs sum to 1"""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = softmax(x)
        assert np.isclose(np.sum(result), 1.0)
    
    def test_softmax_positive_values(self):
        """Test that all softmax outputs are positive"""
        x = np.array([-5.0, 0.0, 5.0])
        result = softmax(x)
        assert np.all(result > 0)
    
    def test_softmax_numerical_stability(self):
        """Test that softmax handles large values without overflow"""
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert np.isclose(np.sum(result), 1.0)
    
    def test_softmax_2d(self):
        """Test softmax on 2D array along last axis"""
        x = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
        result = softmax(x, axis=-1)
        # Each row should sum to 1
        row_sums = np.sum(result, axis=-1)
        assert np.allclose(row_sums, 1.0)


class TestScaledDotProductAttention:
    """Tests for the scaled dot-product attention function"""
    
    def test_basic_attention_shape(self):
        """Test that output shapes are correct"""
        seq_len_q, seq_len_k, d_k, d_v = 3, 4, 8, 16
        
        Q = np.random.randn(seq_len_q, d_k)
        K = np.random.randn(seq_len_k, d_k)
        V = np.random.randn(seq_len_k, d_v)
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (seq_len_q, d_v)
        assert attention_weights.shape == (seq_len_q, seq_len_k)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights for each query sum to 1"""
        Q = np.random.randn(3, 4)
        K = np.random.randn(5, 4)
        V = np.random.randn(5, 6)
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        # Each row (query) should have weights that sum to 1
        row_sums = np.sum(attention_weights, axis=-1)
        assert np.allclose(row_sums, 1.0)
    
    def test_attention_weights_positive(self):
        """Test that all attention weights are non-negative"""
        Q = np.random.randn(3, 4)
        K = np.random.randn(5, 4)
        V = np.random.randn(5, 6)
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        assert np.all(attention_weights >= 0)
    
    def test_batched_attention(self):
        """Test attention with batch dimension"""
        batch_size, seq_len, d_k = 2, 3, 4
        
        Q = np.random.randn(batch_size, seq_len, d_k)
        K = np.random.randn(batch_size, seq_len, d_k)
        V = np.random.randn(batch_size, seq_len, d_k)
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_k)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)
        
        # Check that each batch's attention weights sum correctly
        for b in range(batch_size):
            row_sums = np.sum(attention_weights[b], axis=-1)
            assert np.allclose(row_sums, 1.0)
    
    def test_attention_with_mask(self):
        """Test that masking correctly zeros out attention to masked positions"""
        seq_len, d_k = 4, 8
        
        Q = np.random.randn(seq_len, d_k)
        K = np.random.randn(seq_len, d_k)
        V = np.random.randn(seq_len, d_k)
        
        # Create a causal mask (upper triangular)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        
        # Masked positions should have near-zero attention
        # (numerically, they won't be exactly zero due to softmax)
        masked_positions = attention_weights[mask]
        assert np.all(masked_positions < 1e-6)
    
    def test_dimension_mismatch_error(self):
        """Test that mismatched dimensions raise appropriate errors"""
        Q = np.random.randn(3, 4)
        K = np.random.randn(5, 6)  # Wrong d_k
        V = np.random.randn(5, 8)
        
        with pytest.raises(ValueError, match="same depth"):
            scaled_dot_product_attention(Q, K, V)
    
    def test_kv_length_mismatch_error(self):
        """Test that K and V must have the same sequence length"""
        Q = np.random.randn(3, 4)
        K = np.random.randn(5, 4)
        V = np.random.randn(6, 4)  # Different seq_len than K
        
        with pytest.raises(ValueError, match="same sequence length"):
            scaled_dot_product_attention(Q, K, V)
    
    def test_identity_attention(self):
        """Test a simple case where Q=K and values are one-hot"""
        # When Q=K and they're orthogonal, attention should be mostly on diagonal
        d_k = 4
        seq_len = 3
        
        # Create orthogonal Q and K
        Q = K = np.eye(seq_len, d_k)
        V = np.eye(seq_len)  # Identity matrix for values
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        # Attention should be strongest on the diagonal
        diag_attention = np.diagonal(attention_weights)
        assert np.all(diag_attention > 0.3)  # Reasonable threshold
    
    def test_uniform_keys(self):
        """Test when all keys are identical"""
        seq_len, d_k = 4, 8
        
        Q = np.random.randn(seq_len, d_k)
        K = np.ones((seq_len, d_k))  # All keys identical
        V = np.random.randn(seq_len, d_k)
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        # When all keys are identical, attention should be uniform
        # (each query attends equally to all keys)
        expected_weight = 1.0 / seq_len
        assert np.allclose(attention_weights, expected_weight, rtol=0.1)
    
    def test_scaling_effect(self):
        """Test that scaling by sqrt(d_k) affects the sharpness of attention"""
        # This is more of a conceptual test - we can't directly test the scaling
        # but we can verify the computation is stable
        d_k = 64  # Large dimension
        seq_len = 5
        
        Q = np.random.randn(seq_len, d_k) * 10  # Large values
        K = np.random.randn(seq_len, d_k) * 10
        V = np.random.randn(seq_len, d_k)
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        # Should not have numerical issues
        assert not np.any(np.isnan(output))
        assert not np.any(np.isnan(attention_weights))
        
        # Weights should still sum to 1
        row_sums = np.sum(attention_weights, axis=-1)
        assert np.allclose(row_sums, 1.0)


class TestMultiDimensionalAttention:
    """Tests for attention with additional dimensions (batch, heads, etc.)"""
    
    def test_3d_attention(self):
        """Test with 3D inputs (e.g., batch × seq_len × d_k)"""
        batch, seq_len, d_k = 2, 4, 8
        
        Q = np.random.randn(batch, seq_len, d_k)
        K = np.random.randn(batch, seq_len, d_k)
        V = np.random.randn(batch, seq_len, d_k)
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (batch, seq_len, d_k)
        assert attention_weights.shape == (batch, seq_len, seq_len)
    
    def test_4d_attention(self):
        """Test with 4D inputs (e.g., batch × heads × seq_len × d_k)"""
        batch, heads, seq_len, d_k = 2, 4, 3, 8
        
        Q = np.random.randn(batch, heads, seq_len, d_k)
        K = np.random.randn(batch, heads, seq_len, d_k)
        V = np.random.randn(batch, heads, seq_len, d_k)
        
        output, attention_weights = scaled_dot_product_attention(Q, K, V)
        
        assert output.shape == (batch, heads, seq_len, d_k)
        assert attention_weights.shape == (batch, heads, seq_len, seq_len)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
