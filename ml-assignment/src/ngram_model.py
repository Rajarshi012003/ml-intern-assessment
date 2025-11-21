import random
import re
from collections import defaultdict

class TrigramModel:
    def __init__(self):
        """
        Initializes the TrigramModel.
        """
        # Nested dictionary to store trigram counts: trigram_counts[w1][w2][w3] = count
        self.trigram_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Also store bigram counts for context tracking: bigram_counts[w1][w2] = count
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        
        # Special tokens for sentence boundaries
        self.START_TOKEN = "<START>"
        self.END_TOKEN = "<END>"
        self.UNKNOWN_TOKEN = "<UNK>"
        
        # Vocabulary to track known words
        self.vocabulary = set()
        
        # Flag to check if model has been trained
        self.is_trained = False

    def _clean_text(self, text):
        """
        Cleans the input text by converting to lowercase and normalizing whitespace.
        Preserves sentence boundaries by splitting on periods, exclamation marks, and question marks.
        
        Args:
            text (str): The raw text to clean.
            
        Returns:
            list: List of sentences (each sentence is a string).
        """
        # Convert to lowercase
        text = text.lower()
        
        # Split into sentences based on sentence-ending punctuation
        # Keep the punctuation as part of the sentence
        sentences = re.split(r'([.!?]+)', text)
        
        # Recombine sentences with their punctuation
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if sentences[i].strip():  # Skip empty strings
                sentence = sentences[i].strip()
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]
                combined_sentences.append(sentence)
        
        # Handle the last element if it doesn't have punctuation
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            combined_sentences.append(sentences[-1].strip())
        
        return combined_sentences

    def _tokenize(self, text):
        """
        Tokenizes text into words, removing most punctuation but keeping some meaningful ones.
        
        Args:
            text (str): The text to tokenize.
            
        Returns:
            list: List of tokens (words).
        """
        # Remove most punctuation except apostrophes (for contractions)
        # This regex keeps alphanumeric characters and apostrophes
        text = re.sub(r"[^\w\s']", '', text)
        
        # Split on whitespace and filter out empty strings
        tokens = [token for token in text.split() if token]
        
        return tokens

    def fit(self, text):
        """
        Trains the trigram model on the given text.

        Args:
            text (str): The text to train the model on.
        """
        if not text or not text.strip():
            # Handle empty text case
            self.is_trained = False
            return
        
        # Clean and split text into sentences
        sentences = self._clean_text(text)
        
        # Process each sentence
        for sentence in sentences:
            # Tokenize the sentence
            tokens = self._tokenize(sentence)
            
            if len(tokens) == 0:
                continue
            
            # Add tokens to vocabulary
            self.vocabulary.update(tokens)
            
            # Pad the sentence with start and end tokens
            padded_tokens = [self.START_TOKEN, self.START_TOKEN] + tokens + [self.END_TOKEN]
            
            # Count trigrams and bigrams
            for i in range(len(padded_tokens) - 2):
                w1 = padded_tokens[i]
                w2 = padded_tokens[i + 1]
                w3 = padded_tokens[i + 2]
                
                # Increment trigram count
                self.trigram_counts[w1][w2][w3] += 1
                
                # Increment bigram count (for context)
                self.bigram_counts[w1][w2] += 1
        
        self.is_trained = True

    def _sample_next_word(self, w1, w2):
        """
        Probabilistically samples the next word given the previous two words.
        
        Args:
            w1 (str): The first word of the context.
            w2 (str): The second word of the context.
            
        Returns:
            str: The sampled next word.
        """
        # Get all possible next words and their counts
        possible_words = self.trigram_counts[w1][w2]
        
        if not possible_words:
            # If no trigram found, return END_TOKEN to stop generation
            return self.END_TOKEN
        
        # Convert counts to probabilities
        total_count = sum(possible_words.values())
        words = list(possible_words.keys())
        probabilities = [possible_words[word] / total_count for word in words]
        
        # Sample from the distribution
        sampled_word = random.choices(words, weights=probabilities, k=1)[0]
        
        return sampled_word

    def generate(self, max_length=50, seed=None):
        """
        Generates new text using the trained trigram model.

        Args:
            max_length (int): The maximum length of the generated text (in words).
            seed (int, optional): Random seed for reproducibility.

        Returns:
            str: The generated text.
        """
        if not self.is_trained:
            return ""
        
        if seed is not None:
            random.seed(seed)
        
        # Start with two START tokens
        w1 = self.START_TOKEN
        w2 = self.START_TOKEN
        generated_words = []
        
        # Generate words
        for _ in range(max_length):
            # Sample the next word
            next_word = self._sample_next_word(w1, w2)
            
            # Stop if we hit the END token
            if next_word == self.END_TOKEN:
                break
            
            # Add the word to our generated text
            generated_words.append(next_word)
            
            # Update context window
            w1 = w2
            w2 = next_word
        
        # Join the words and return
        return ' '.join(generated_words)
