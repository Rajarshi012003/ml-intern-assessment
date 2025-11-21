import os
import sys
from ngram_model import TrigramModel

def main():
    # Create a new TrigramModel
    model = TrigramModel()
    
    # Determine the path to Alice in Wonderland text file
    # The file is in the parent directory of ml-intern-assessment-main
    # __file__ is in ml-assignment/src/generate.py
    # So we need to go up: src -> ml-assignment -> ml-intern-assessment-main -> parent
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    alice_path = os.path.join(base_path, "Alice in Wonderland.txt")
    
    # Check if the file exists
    if not os.path.exists(alice_path):
        print(f"Error: Could not find 'Alice in Wonderland.txt' at {alice_path}")
        print("Please ensure the file is in the correct location.")
        sys.exit(1)
    
    print("Loading and training on Alice in Wonderland...")
    print(f"Reading from: {alice_path}")
    
    # Train the model on Alice in Wonderland
    with open(alice_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    print(f"Text loaded: {len(text)} characters")
    print("Training model...")
    model.fit(text)
    print("Training complete!")
    
    # Generate new text with different lengths
    print("\n" + "="*60)
    print("Generated Text Sample 1 (max 30 words):")
    print("="*60)
    generated_text_1 = model.generate(max_length=30)
    print(generated_text_1)
    
    print("\n" + "="*60)
    print("Generated Text Sample 2 (max 50 words):")
    print("="*60)
    generated_text_2 = model.generate(max_length=50)
    print(generated_text_2)
    
    print("\n" + "="*60)
    print("Generated Text Sample 3 (max 70 words):")
    print("="*60)
    generated_text_3 = model.generate(max_length=70)
    print(generated_text_3)
    
    print("\n" + "="*60)
    print("Model Statistics:")
    print("="*60)
    print(f"Vocabulary size: {len(model.vocabulary)} unique words")
    print(f"Number of unique bigram contexts: {len(model.bigram_counts)}")
    total_trigrams = 0
    for w1_dict in model.trigram_counts.values():
        for w2_dict in w1_dict.values():
            total_trigrams += sum(w2_dict.values())
    print(f"Total trigrams counted: {total_trigrams}")

if __name__ == "__main__":
    main()
