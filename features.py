import pandas as pd
import re
from nltk.featstruct import FeatStruct

try:
    df = pd.read_csv('VNC2013.csv')
except:
    print('VNC2013.csv not found!')

def wordshape(text):
    t1 = re.sub('[A-Z]', 'X',text)
    t2 = re.sub('[a-z]', 'x', t1)
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

def capital_features(sentence, i, history):
    word, pos = sentence[i]
    capital = sum(1 for c in word if c.isupper())
    return {
        "capital": capital,
        "pos": pos,
        "whole history": tuple(history)
    }

def len_features(sentence, i, history):
    word, pos = sentence[i]
    capital = sum(1 for c in word if c.isupper())
    length = len(word)
    return {
        "capital": capital,
        "length": length,
        "pos": pos,
        "whole history": tuple(history)
    }

def name_features(sentence, i, history):
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

def punc_features(sentence, i, history): # does nothing special :D
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

def capital2_features(sentence, i, history): # this is not good, uh precision is very much lower help bye good
    word, pos = sentence[i]
    isInList = True if word in set(df['Name']) else False
    containsPunc = '.' or '-' in word
    firstLetterCapital = word[0].isupper()
    consecutiveCapitals = 0
    for i in range(len(word)):
        if word[i].isupper():
            consecutiveCapitals += 1
        else:
            break
    length = len(word)
    return {
        "isInList": isInList,
        "containsPunc": containsPunc,
        "firstLetterCapital": firstLetterCapital,
        "consecutiveCapitals": consecutiveCapitals,
        "length": length,
        "pos": pos,
        "whole history": tuple(history)
    }

def wordshape_features(sentence, i, history):
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

def number_features(sentence, i, history): # this is not good, uh precision is very much lower help bye good
    word, pos = sentence[i]
    isInList = True if word in set(df['Name']) else False
    containsPunc = '.' or '-' in word
    firstLetterCapital = word[0].isupper()
    containsNumbers = any(char.isdigit() for char in word)
    consecutiveCapitals = 0
    for i in range(len(word)):
        if word[i].isupper():
            consecutiveCapitals += 1
        else:
            break
    length = len(word)
    return {
        "isInList": isInList,
        "containsPunc": containsPunc,
        "firstLetterCapital": firstLetterCapital,
        "containsNumbers": containsNumbers,
        "consecutiveCapitals": consecutiveCapitals,
        "length": length,
        "pos": pos,
        "whole history": tuple(history)
    }

def big_features(sentence, i, history):
    word, pos = sentence[i]

    features = {
        'bias': 1.0,
        'without-caps': word.lower(),
        'in-list': True if word in set(df['Name']) else False,
        'word[-3:]': word[-3:],
        'all-caps': word.isupper(),
        'title': word.istitle(),
        'all-digits': word.isdigit(),
        'contains-dot': '.' in word,
        'contains-hyphen': '-' in word,
        'wordshape': wordshape(word),
        'pos': pos,
        'history': tuple(history)
    }
    if i > 0:
        w_m, p_m = sentence[i -1]
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

    if i < len(sentence)-1:
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
    "capital_features": capital_features,
    "len_features": len_features,
    "name_features": name_features,
    "punc_features": punc_features,
    "capital2_features": capital2_features,
    "number_features": number_features,
    "wordshape_features": wordshape_features,
    "big_features": big_features,
}