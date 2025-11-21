# Trigram Language Model

This directory contains the implementation of a Trigram (N=3) Language Model trained on "Alice's Adventures in Wonderland" by Lewis Carroll.

## Project Structure

```
ml-assignment/
├── src/
│   ├── ngram_model.py      # Core TrigramModel implementation
│   ├── generate.py          # Script to train and generate text
│   └── utils.py            # Utility functions for data loading
├── tests/
│   └── test_ngram.py       # Unit tests for the model
├── data/
│   └── example_corpus.txt  # Small example corpus
├── README.md               # This file
└── evaluation.md           # Design choices documentation
```

## Requirements

- Python 3.7+
- pytest (for running tests)

Install dependencies:
```bash
pip install -r ../requirements.txt
```

## How to Run

### 1. Run Unit Tests

To verify the implementation passes all test cases:

```bash
cd ml-assignment
python -m pytest tests/test_ngram.py -v
```

Expected output:
```
tests/test_ngram.py::test_fit_and_generate PASSED
tests/test_ngram.py::test_empty_text PASSED
tests/test_ngram.py::test_short_text PASSED
```

### 2. Generate Text from Alice in Wonderland

To train the model on Alice in Wonderland and generate sample text:

```bash
cd ml-assignment/src
python generate.py
```

This will:
- Load and process the "Alice in Wonderland.txt" file
- Train the trigram model on the entire text
- Generate 3 sample texts of varying lengths (30, 50, 70 words)
- Display model statistics (vocabulary size, trigram counts, etc.)

### 3. Use the Model Programmatically

You can also use the `TrigramModel` class directly in your Python code:

```python
from src.ngram_model import TrigramModel

# Create and train the model
model = TrigramModel()
with open("path/to/text.txt", "r") as f:
    text = f.read()
model.fit(text)

# Generate new text
generated = model.generate(max_length=50)
print(generated)
```

## File Locations

**Important:** The `generate.py` script expects to find "Alice in Wonderland.txt" in the parent directory of `ml-intern-assessment-main`:

```
/path/to/parent/
├── Alice in Wonderland.txt
└── ml-intern-assessment-main/
    └── ml-assignment/
```

If your file is located elsewhere, update the path in `generate.py` line 11-16.

## Model Features

- **Text Cleaning**: Converts to lowercase, removes unnecessary punctuation
- **Sentence Segmentation**: Processes text sentence-by-sentence
- **Special Tokens**: Uses `<START>` and `<END>` tokens for proper boundaries
- **Probabilistic Generation**: Samples next words based on learned probability distributions
- **Efficient Storage**: Uses nested defaultdict for O(1) trigram lookups

## Design Choices

For a detailed explanation of the implementation and design decisions, please see `evaluation.md`.

## Model Statistics (Alice in Wonderland)

When trained on the full Alice in Wonderland text:
- **Vocabulary Size**: 2,860 unique words
- **Unique Bigram Contexts**: 2,683
- **Total Trigrams Counted**: 28,143
- **Text Size**: ~145,000 characters

## Example Generated Text

```
then they both sat silent and looked at two
```

```
what are they made of solid glass there was certainly english
```

Note: Generated text is probabilistic and will vary with each run.
