import pandas as pd

try:
    df = pd.read_csv('VNC2013.csv')
except:
    print('VNC2013.csv not found!')

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



functions = {
    "test_features": test_features,
    "capital_features": capital_features,
    "len_features": len_features,
    "name_features": name_features,
    "punc_features": punc_features,
    "capital2_features": capital2_features,
    "number_features": number_features,
}