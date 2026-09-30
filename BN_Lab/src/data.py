"""
data.py -- training data and special tokens for the lab.

The dataset is the small list of sentences given in the lab handout.
Every sentence is lower-cased and split into word tokens (one word = one token).
The special tokens <START> and <END> mark the beginning and end of a sentence;
they are added later, inside the models, so this file only holds the plain words.
"""

# Special tokens. <START> is only ever used as context, <END> is only ever predicted.
START = "<START>"
END = "<END>"

# The six training sentences from the lab handout.
RAW_TEXT = """
the cat sat on the mat
the cat sat on the rug
the dog sat on the mat
the dog ran to the park
the cat ran to the park
the dog sat on the rug
"""


def load_sentences():
    """Return the training data as a list of sentences, each a list of words.

    Example: [["the", "cat", "sat", "on", "the", "mat"], ...]
    """
    sentences = []
    for line in RAW_TEXT.strip().splitlines():
        sentences.append(line.lower().split())   # lower-case, then split on spaces
    return sentences
