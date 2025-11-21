# This file contains utility functions for data loading and preprocessing.

import os

def load_text_file(filepath):
    """
    Loads a text file and returns its contents.
    
    Args:
        filepath (str): Path to the text file.
        
    Returns:
        str: The contents of the file.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def get_alice_path():
    """
    Returns the path to the Alice in Wonderland text file.
    Searches in common locations relative to the project structure.
    
    Returns:
        str: Path to Alice in Wonderland.txt
        
    Raises:
        FileNotFoundError: If the file cannot be found.
    """
    # Try multiple possible locations
    possible_paths = [
        # Relative to src directory
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                     "Alice in Wonderland.txt"),
        # Relative to ml-assignment directory
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                     "..", "..", "Alice in Wonderland.txt"),
        # Direct path
        "Alice in Wonderland.txt",
    ]
    
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            return normalized_path
    
    raise FileNotFoundError(
        "Could not find 'Alice in Wonderland.txt'. "
        "Please ensure it's in the project root directory."
    )
