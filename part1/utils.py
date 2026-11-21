import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import zlib

random.seed(0)

### Simulate keyboard typos
TYPO_PROB = 0.35
MIN_WORD_LEN = 3
TYPO_SEED = 0

KEY_ROWS = [("qwertyuiop", 0.0), ("asdfghjkl", 0.25), ("zxcvbnm", 0.75)]
KEY_POS = {k: (row, col + shift) for row, (keys, shift) in enumerate(KEY_ROWS) for col, k in enumerate(keys)}
# Neighbours (e.g. s -> a, d, w, e, z, x)
NEIGHBORS = {
    k: [m for m, (r2, c2) in KEY_POS.items() if m != k and (r - r2) ** 2 + (c - c2) ** 2 <= 1.3 ** 2]
    for k, (r, c) in KEY_POS.items()
}


def typo(word, rng):
    """Apply one keyboard typo to a word"""
    i = rng.randrange(1, len(word))  # position of the typo
    op = rng.choice(("substitute", "delete", "insert", "swap"))
    ch = word[i]

    if op == "swap":  # two neighbouring letters typed in the wrong order
        a = i if i + 1 < len(word) else i - 1 
        if word[a] != word[a + 1]:
            return word[:a] + word[a + 1] + word[a] + word[a + 2:]
        op = "substitute"

    if op == "delete":  # a key not pressed
        return word[:i] + word[i + 1:]

    near = rng.choice(NEIGHBORS[ch.lower()])
    near = near.upper() if ch.isupper() else near
    if op == "insert":  # the neighbouring key is hit as well
        return word[:i + 1] + near + word[i + 1:]
    return word[:i] + near + word[i + 1:]  # the neighbouring key is hit instead


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    # setting SEED for reproducibility
    rng = random.Random(TYPO_SEED * 2 ** 32 + zlib.crc32(text.encode("utf-8")))

    def maybetypo(match):
        word = match.group()
        return typo(word, rng) if rng.random() < TYPO_PROB else word

    # change only letters
    example["text"] = re.sub(r"[A-Za-z]{%d,}" % MIN_WORD_LEN, maybetypo, text)

    ##### YOUR CODE ENDS HERE ######

    return example
