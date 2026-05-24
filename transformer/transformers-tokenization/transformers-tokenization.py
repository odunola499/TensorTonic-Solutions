import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        words = []
        for text in texts:
            splits = text.lower().split(' ')
            words += splits
        words = sorted(list(set(words)))

        for index, word in enumerate(words):
            self.word_to_id[word] = index + 4

        self.id_to_word = {j:i for i,j in self.word_to_id.items()}
        self.vocab_size =len(self.id_to_word)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        if len(text) == 0:
            return []
        words = text.split(' ')
        ids = [self.word_to_id[word.lower()] if word.lower() in self.word_to_id else 1 for word in words]
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = [self.id_to_word[id] if id in self.id_to_word else "<UNK>" for id in ids]
        return ' '.join(words)
