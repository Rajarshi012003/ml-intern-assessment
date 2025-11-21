"""
Demonstration of Scaled Dot-Product Attention

This script demonstrates the scaled_dot_product_attention function with various
examples to show how it works in different scenarios.
"""

import numpy as np
import sys
from attention import scaled_dot_product_attention, visualize_attention, attention_summary


def demo_1_simple_attention():
    """
    Demo 1: Simple 3x3 attention example
    Shows basic attention mechanism with small matrices
    """
    print("=" * 80)
    print("DEMO 1: Simple Attention (3 queries, 3 keys, 3 values)")
    print("=" * 80)
    
    # Create simple Q, K, V matrices
    # Shape: (seq_len, d_k) where seq_len=3, d_k=4
    np.random.seed(42)
    Q = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0]
    ])
    
    K = np.array([
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.5, 0.5, 0.5, 0.5]
    ])
    
    V = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ])
    
    print("\nQuery matrix Q (3x4):")
    print(Q)
    print("\nKey matrix K (3x4):")
    print(K)
    print("\nValue matrix V (3x3):")
    print(V)
    
    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)
    
    visualize_attention(attention_weights, 
                       query_labels=["Q1", "Q2", "Q3"],
                       key_labels=["K1", "K2", "K3"])
    
    print("\nOutput matrix (3x3):")
    print(output)
    
    print("\nAttention Summary:")
    summary = attention_summary(attention_weights)
    for key, value in summary.items():
        if key != 'row_sums':
            print(f"  {key}: {value}")
    print(f"  row_sums (should be ~1.0): {summary['row_sums']}")
    
    print("\n" + "=" * 80 + "\n")


def demo_2_with_mask():
    """
    Demo 2: Attention with masking
    Shows how masking prevents attention to certain positions (e.g., future tokens)
    """
    print("=" * 80)
    print("DEMO 2: Attention with Causal Masking (prevents looking ahead)")
    print("=" * 80)
    
    np.random.seed(42)
    seq_len = 4
    d_k = 8
    
    # Random Q, K, V
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    
    print(f"\nSequence length: {seq_len}, Hidden dimension: {d_k}")
    
    # Create causal mask (lower triangular)
    # This prevents each position from attending to future positions
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    
    print("\nCausal mask (True = masked, False = visible):")
    print(mask.astype(int))
    print("(Diagonal and below = 0 [can attend], Above diagonal = 1 [masked])")
    
    # Compute attention without mask
    output_no_mask, attention_no_mask = scaled_dot_product_attention(Q, K, V, mask=None)
    
    # Compute attention with mask
    output_with_mask, attention_with_mask = scaled_dot_product_attention(Q, K, V, mask=mask)
    
    print("\n" + "-" * 80)
    print("WITHOUT MASK:")
    print("-" * 80)
    visualize_attention(attention_no_mask,
                       query_labels=[f"Pos{i}" for i in range(seq_len)],
                       key_labels=[f"Pos{i}" for i in range(seq_len)])
    
    print("\n" + "-" * 80)
    print("WITH CAUSAL MASK:")
    print("-" * 80)
    visualize_attention(attention_with_mask,
                       query_labels=[f"Pos{i}" for i in range(seq_len)],
                       key_labels=[f"Pos{i}" for i in range(seq_len)])
    
    print("\nNote: With causal mask, each position can only attend to itself and previous positions.")
    print("The attention weights for future positions (upper triangle) are ~0.")
    
    print("\n" + "=" * 80 + "\n")


def demo_3_batch_attention():
    """
    Demo 3: Batch attention
    Shows how the function handles batched inputs (multiple sequences at once)
    """
    print("=" * 80)
    print("DEMO 3: Batch Attention (processing multiple sequences simultaneously)")
    print("=" * 80)
    
    np.random.seed(42)
    batch_size = 2
    seq_len = 3
    d_k = 4
    d_v = 4
    
    # Create batched Q, K, V
    # Shape: (batch_size, seq_len, d_k/d_v)
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_v)
    
    print(f"\nBatch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Key/Query dimension: {d_k}")
    print(f"Value dimension: {d_v}")
    
    print(f"\nQ shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    
    # Compute attention for the batch
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    print("\n" + "-" * 80)
    print("Attention weights for Batch 0:")
    print("-" * 80)
    visualize_attention(attention_weights[0])
    
    print("\n" + "-" * 80)
    print("Attention weights for Batch 1:")
    print("-" * 80)
    visualize_attention(attention_weights[1])
    
    print("\n" + "=" * 80 + "\n")


def demo_4_multi_head_attention():
    """
    Demo 4: Simulating Multi-Head Attention
    Shows how scaled dot-product attention can be used in a multi-head setup
    """
    print("=" * 80)
    print("DEMO 4: Multi-Head Attention (multiple attention heads in parallel)")
    print("=" * 80)
    
    np.random.seed(42)
    num_heads = 4
    seq_len = 5
    d_model = 8  # Total dimension
    d_k = d_model // num_heads  # Dimension per head
    
    print(f"\nNumber of attention heads: {num_heads}")
    print(f"Sequence length: {seq_len}")
    print(f"Model dimension (d_model): {d_model}")
    print(f"Dimension per head (d_k): {d_k}")
    
    # Create Q, K, V for multi-head attention
    # Shape: (num_heads, seq_len, d_k)
    Q = np.random.randn(num_heads, seq_len, d_k)
    K = np.random.randn(num_heads, seq_len, d_k)
    V = np.random.randn(num_heads, seq_len, d_k)
    
    print(f"\nQ shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    
    # Compute attention for all heads simultaneously
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    print("\n" + "-" * 80)
    print("Attention patterns vary across heads:")
    print("-" * 80)
    
    for head in range(min(2, num_heads)):  # Show first 2 heads
        print(f"\nHead {head}:")
        visualize_attention(attention_weights[head])
    
    print("\nIn a real multi-head attention layer, these outputs would be concatenated")
    print("and linearly transformed to produce the final output.")
    
    print("\n" + "=" * 80 + "\n")


def demo_5_sentence_attention():
    """
    Demo 5: Realistic sentence example
    Demonstrates attention on a simple sentence with interpretable tokens
    """
    print("=" * 80)
    print("DEMO 5: Sentence-Level Attention (Interpretable Example)")
    print("=" * 80)
    
    # Simulated sentence: "The cat sat on mat"
    tokens = ["The", "cat", "sat", "on", "mat"]
    seq_len = len(tokens)
    d_k = 4
    
    print(f"\nSentence: {' '.join(tokens)}")
    print(f"Number of tokens: {seq_len}")
    
    # Create meaningful Q, K, V matrices
    # Let's make "cat" and "mat" similar (they're both nouns)
    # and "sat" and "on" similar (they relate to position)
    np.random.seed(42)
    
    # Simplified embeddings (in reality, these would come from a trained model)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    
    # Make "cat" and "mat" attend to each other more
    Q[1] = K[4] + np.random.randn(d_k) * 0.1  # cat queries similar to mat key
    Q[4] = K[1] + np.random.randn(d_k) * 0.1  # mat queries similar to cat key
    
    print("\nQ, K, V matrices created with some semantic similarity between 'cat' and 'mat'")
    
    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    print("\n" + "-" * 80)
    print("ATTENTION WEIGHTS:")
    print("-" * 80)
    visualize_attention(attention_weights, query_labels=tokens, key_labels=tokens)
    
    print("\nInterpretation:")
    print("- Each row shows which words a token attends to")
    print("- Higher values mean stronger attention")
    print("- Notice how 'cat' and 'mat' might have higher mutual attention")
    
    print("\n" + "=" * 80 + "\n")


def main():
    """Run all demonstrations"""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "  SCALED DOT-PRODUCT ATTENTION - COMPREHENSIVE DEMONSTRATION".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\n")
    
    # Run all demos
    demo_1_simple_attention()
    demo_2_with_mask()
    demo_3_batch_attention()
    demo_4_multi_head_attention()
    demo_5_sentence_attention()
    
    print("*" * 80)
    print("*" + "  ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY".center(78) + "*")
    print("*" * 80)
    print("\n")
    print("Key Takeaways:")
    print("1. Attention allows each query to focus on relevant keys/values")
    print("2. Scaling by sqrt(d_k) stabilizes gradients")
    print("3. Masking enables causal (autoregressive) attention")
    print("4. The mechanism naturally handles batching and multi-head attention")
    print("5. Attention weights sum to 1 (valid probability distribution)")


if __name__ == "__main__":
    main()
