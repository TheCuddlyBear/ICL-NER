"""
FILE: best_model_test.py
Author: Alexis Dimitriadis

Tests the functionality of a pickled model called best.pickle
"""

import pickle
from nltk.corpus import conll2002 as conll

# best.pickle
ner = pickle.load(open("best.pickle", "rb"))

# Usage 1: parse a list of sentences (with POS tags)
tagzinnen = conll.tagged_sents("ned.train")[1000:1050]
ner.parse_sents(tagzinnen)

# Usage 2: self-evaluate (on chunked sentences)
chunkzinnen = conll.chunked_sents("ned.testa")[1000:1500]

print(ner.accuracy(chunkzinnen))