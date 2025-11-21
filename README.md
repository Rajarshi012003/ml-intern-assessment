#AI/ML Intern Assignment - COMPLETED ✓

Welcome to the AI/ML Intern assignment! This project is designed to test your core Python skills and your ability to design and build a clean and efficient system from scratch.

## ✅ Implementation Status

**Task 1: Trigram Language Model** - **COMPLETED** ✓
- ✓ Fully implemented TrigramModel class with proper n-gram counting
- ✓ Text cleaning, tokenization, and sentence segmentation
- ✓ Probabilistic text generation with random sampling
- ✓ Trained on Alice's Adventures in Wonderland
- ✓ All unit tests passing (3/3)
- ✓ Comprehensive documentation

**Task 2: Scaled Dot-Product Attention** - **COMPLETED** ✓ (Optional)
- ✓ Pure NumPy implementation of attention mechanism
- ✓ Numerically stable softmax function
- ✓ Support for masking (causal, padding)
- ✓ Handles batching and multi-head attention
- ✓ All unit tests passing (16/16)
- ✓ 5 comprehensive demonstration scripts
- ✓ Complete documentation and visualization tools

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
cd ml-assignment
python -m pytest tests/test_ngram.py -v
```

### 3. Generate Text
```bash
cd ml-assignment/src
python generate.py
```

## Implementation Overview

### Files Created/Modified

**Task 1: Trigram Language Model**
- `ml-assignment/src/ngram_model.py` - Complete TrigramModel implementation (200+ lines)
- `ml-assignment/src/generate.py` - Updated to use Alice in Wonderland text
- `ml-assignment/src/utils.py` - Utility functions for data loading
- `ml-assignment/evaluation.md` - Comprehensive design documentation
- `ml-assignment/README.md` - Detailed usage instructions
- `ml-assignment/tests/test_ngram.py` - Unit tests (3 tests)
- Added `__init__.py` files for proper Python package structure

**Task 2: Scaled Dot-Product Attention**
- `attention-task/src/attention.py` - Complete attention implementation (250+ lines)
- `attention-task/src/demo.py` - 5 comprehensive demonstrations (350+ lines)
- `attention-task/tests/test_attention.py` - Unit tests (16 tests)
- `attention-task/README.md` - Full documentation with examples
- Added `__init__.py` files for proper package structure

### Key Features
1. **Nested Dictionary Storage**: Efficient O(1) lookup for trigram counts
2. **Probabilistic Sampling**: Uses `random.choices()` with proper probability weights
3. **Sentence Segmentation**: Processes text sentence-by-sentence for better structure
4. **Special Tokens**: `<START>` and `<END>` tokens for proper boundaries
5. **Comprehensive Testing**: All test cases pass successfully

### Model Statistics
When trained on Alice in Wonderland:
- **Vocabulary**: 2,860 unique words
- **Trigrams**: 28,143 total count
- **Bigram Contexts**: 2,683 unique pairs

## Documentation

- **Design Choices**: See `ml-assignment/evaluation.md` for detailed design decisions
- **Usage Instructions**: See `ml-assignment/README.md` for how to run the code
- **Assignment Details**: See `assignment.md` for original requirements

## Example Output

```
Generated Text Sample:
its really dreadful she muttered to herself thats quite enough i hope i shant 
grow any more as it is a raven like a telescope
```

## Testing

### Task 1: Trigram Language Model Tests

All tests pass successfully:
```bash
cd ml-assignment
python -m pytest tests/test_ngram.py -v
```

```
tests/test_ngram.py::test_fit_and_generate PASSED
tests/test_ngram.py::test_empty_text PASSED
tests/test_ngram.py::test_short_text PASSED

3 passed in 0.01s
```

### Task 2: Scaled Dot-Product Attention Tests

All tests pass successfully:
```bash
cd attention-task
python -m pytest tests/test_attention.py -v
```

```
16 passed in 0.16s
```

Tests cover:
- Softmax correctness and numerical stability
- Attention shape validation
- Probability distribution properties
- Masking functionality
- Batch and multi-head processing
- Error handling

## Running Demonstrations

### Task 1: Generate Text from Trigram Model
```bash
cd ml-assignment/src
python generate.py
```

Sample output:
```
Generated Text Sample:
its really dreadful she muttered to herself thats quite enough i hope i shant 
grow any more as it is a raven like a telescope
```

### Task 2: Attention Mechanism Demos
```bash
cd attention-task/src
python demo.py
```

This runs 5 comprehensive demonstrations:
1. Simple 3×3 attention
2. Causal masking (autoregressive models)
3. Batch processing
4. Multi-head attention
5. Sentence-level attention with interpretable results

## Project Structure

```
ml-intern-assessment-main/
├── ml-assignment/           # Task 1: Trigram Language Model
│   ├── src/
│   │   ├── ngram_model.py  # Core implementation
│   │   ├── generate.py     # Text generation script
│   │   └── utils.py        # Helper functions
│   ├── tests/
│   │   └── test_ngram.py   # Unit tests
│   ├── data/
│   │   └── example_corpus.txt
│   ├── README.md           # Task 1 documentation
│   └── evaluation.md       # Design choices (both tasks)
│
├── attention-task/         # Task 2: Scaled Dot-Product Attention
│   ├── src/
│   │   ├── attention.py    # Core implementation
│   │   └── demo.py         # Demonstrations
│   ├── tests/
│   │   └── test_attention.py  # Unit tests
│   └── README.md           # Task 2 documentation
│
├── README.md               # This file (main overview)
├── assignment.md           # Original assignment
├── quick_start.md          # Quick start guide
└── requirements.txt        # Python dependencies
```

## Next Steps

To review or extend the implementation:

**Task 1 (Trigram Model):**
1. Review design documentation in `ml-assignment/evaluation.md`
2. Explore code in `ml-assignment/src/ngram_model.py`
3. Try modifying hyperparameters (e.g., max_length)
4. Experiment with different text corpora

**Task 2 (Attention):**
1. Review attention documentation in `attention-task/README.md`
2. Explore code in `attention-task/src/attention.py`
3. Run demos to see attention in action
4. Experiment with different Q, K, V matrices

For more detailed instructions, please refer to the `assignment.md` file.
