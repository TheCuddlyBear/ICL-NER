import pandas as pd
import re
from nltk.featstruct import FeatStruct

try:
    df = pd.read_csv('VNC2013.csv')
except:
    print('VNC2013.csv not found!')


def wordshape(text):
    """
    :param text: The word from the sentence
    :return: The word in format (Ccd)

    This function turns all capital letters in to a C, all lowercase letters into a c, and all digits into a d.
    This way we  have a very simple word-shape to compare all words on a basic structure level.
    """
    t1 = re.sub('[A-Z]', 'C', text)
    t2 = re.sub('[a-z]', 'c', t1)
    return re.sub('[0-9]', 'd', t2)


def test_features(sentence, i, history):
    """dummy Chunker features designed to test the Chunker class for correctness
        - the POS tag of the word
        - the entire history of IOB tags so far
            formatted as a tuple because it needs to be hashable
    """
    word, pos = sentence[i]
    return {
        "pos": pos,
        "whole history": tuple(history)
    }


def name_features(sentence, i, history):
    """
        :param sentence: List of (word, pos) tuples
        :param i: Index of words in sentence
        :param history: List of previous IOB tags
        :return: Feature dictionary

        This function looks at 5 features:
       - If the word is in the list of most common Dutch names.
       - If the word contains any capital letters
       - The word length
       - The POS-tags
       - History of previous IOB tags in the sentence

        """
    word, pos = sentence[i]
    isInList = True if word in set(df['Name']) else False
    capital = sum(1 for c in word if c.isupper())
    length = len(word)
    return {
        "isInList": isInList,
        "capital": capital,
        "length": length,
        "pos": pos,
        "whole history": tuple(history)
    }


def punc_features(sentence, i, history):
    """
        :param sentence: List of (word, pos) tuples
        :param i: Index of words in sentence
        :param history: List of previous IOB tags
        :return: Feature dictionary

        This function looks at 6 features:
       - If the word is in the list of most common Dutch names.
       - If the word contains punctuation
       - If the word contains any capital letters
       - The word length
       - The POS-tags
       - History of previous IOB tags in the sentence

        """
    word, pos = sentence[i]
    isInList = True if word in set(df['Name']) else False
    containsPunc = '.' or '-' in word
    capital = sum(1 for c in word if c.isupper())
    length = len(word)
    return {
        "isInList": isInList,
        "containsPunc": containsPunc,
        "capital": capital,
        "length": length,
        "pos": pos,
        "whole history": tuple(history)
    }

def wordshape_features(sentence, i, history):
    """
       :param sentence: List of (word, pos) tuples
       :param i: Index of words in sentence
       :param history: List of previous IOB tags
       :return: Feature dictionary

       This function looks at 5 features:
       - If the word is in a list of most common Dutch names
       - Shape of the word (Ccd)
       - The word length
       - The POS-tags
       - History of previous IOB tags in the sentence

       """
    word, pos = sentence[i]
    isInNameList = True if word in set(df['Name']) else False
    shape = wordshape(word)
    length = len(word)
    return {
        "isInNameList": isInNameList,
        "shape": shape,
        "length": length,
        "pos": pos,
        "whole history": tuple(history)
    }



def big_features(sentence, i, history):
    """
    Feature function: Big Features
    :param sentence: List of (word, pos) tuples
    :param i: Index of words in sentence
    :param history: List of previous IOB tags
    :return: Feature dictionary

    This function uses a window of 2 words, which means it looks to the given word, the word before
    and the word after. This function looks at 11 features:
        - The word in lowercase
        - If the word is in a list of common Dutch names
        - The suffix of the word
        - If the word is all caps
        - If the word starts with a capital letter
        - If it contains a dot (.)
        - If it contains a hyphen (-)
        - Shape of the word (Ccd)
        - The POS-tags
        - History of previous IOB tags in the sentence

    """
    word, pos = sentence[i]

    features = {
        'without-caps': word.lower(),
        'in-list': True if word in set(df['Name']) else False,
        'word[-3:]': word[-3:],
        'all-caps': word.isupper(),
        'title': word.istitle(),
        'all-digits': word.isdigit(),
        'contains-dot': '.' in word,
        'contains-hyphen': '-' in word,
        #'wordshape': wordshape(word),
        'pos': pos,
        'history': tuple(history)
    }
    if i > 0:
        w_m, p_m = sentence[i - 1]
        features.update({
            '-:without-caps': w_m.lower(),
            '-:title': w_m.istitle(),
            '-:all-caps': w_m.isupper(),
            '-:contains-dot': '.' in w_m,
            '-:contains-hyphen': '-' in w_m,
            '-:wordshape': wordshape(w_m),
            '-:pos': p_m,
        })
    else:
        features['begin'] = True

    if i < len(sentence) - 1:
        w_p, p_p = sentence[i + 1]
        features.update({
            '+:without-caps': w_p.lower(),
            '+:title': w_p.istitle(),
            '+:all-caps': w_p.isupper(),
            '+:contains-dot': '.' in w_p,
            '+:contains-hyphen': '-' in w_p,
            '+:wordshape': wordshape(w_p),
            '+:pos': p_p,
        })
    else:
        features['end'] = True

    return features



functions = {
    "test_features": test_features,
    "name_features": name_features,
    "punc_features": punc_features,
    "wordshape_features": wordshape_features,
    "big_features": big_features,
}
