import pickle
import sys
from nltk.corpus import conll2002 as conll
from nltk.corpus import nonbreaking_prefixes
from nltk.corpus import gazetteers

def main():
    print(gazetteers.words("nl"))

if __name__ == "__main__":
    main()