import getopt
import pickle
import sys
from features import *
from nltk.corpus import conll2002 as conll
from custom_chunker import ConsecutiveNPChunker


def createmodel(dataset, feature_function, output, algorithm):
    training = conll.chunked_sents(dataset)
    model = ConsecutiveNPChunker(functions[feature_function], training, algorithm)
    pickleModel(model, output)

def pickleModel(model, output):
    pickle.dump(model, open(output, "wb"))

def main(argv):
    algorithm = ''
    dataset = ''
    feature_function = ''
    output = ''
    try:
        opts, args = getopt.getopt(argv, "hd:a:f:o:", ["dataset=", "algorithm=", "feature_function=", "output="])
    except getopt.GetoptError:
        print(f'Invalid option! Usage: build_models.py -o <outfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('build_models.py -o <outfile>')
            sys.exit()
        elif opt in ('-d', "--dataset"):
            if arg in ("ned.train", "esp.train"):
                dataset = arg
            else:
                print("Dataset should be ned.train or esp.train")
                sys.exit(2)
        elif opt in ("-a", "--algorithm"):
            if arg in ("NaiveBayes", "DecisionTree", "IIS", "GIS"):
                algorithm = arg
            else:
                print(
                    'Invalid algorithm! Choose from the following: \"NaiveBayes\", \"DecisionTree\", \"IIS\", and \"GIS\"')
                sys.exit(2)
        elif opt in ('-f', '--feature_function'):
            feature_function = arg
        elif opt in ("-o", "--output"):
            output = arg

    createmodel(dataset, feature_function, output, algorithm)

if __name__ == "__main__":
    main(sys.argv[1:])
