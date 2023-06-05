import pickle
import sys
from nltk.corpus import conll2002 as conll

def main():
    arg = conll.chunked_sents('ned.train')[1]
    ner = pickle.load(open("capital.pickle", "rb"))
    print(ner.parse(arg))

if __name__ == "__main__":
    main()