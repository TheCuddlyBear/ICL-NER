import pickle
import sys
from nltk.corpus import conll2002 as conll
from nltk.corpus import nonbreaking_prefixes

def main():
    print(nonbreaking_prefixes.words('dutch'))

if __name__ == "__main__":
    main()