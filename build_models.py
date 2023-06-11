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
        opts, args = getopt.getopt(argv, "hd:a:f:o:v", ["dataset=", "algorithm=", "feature=", "output="])
    except getopt.GetoptError:
        print(f'Invalid option! Usage: build_models.py -o <outfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('Model Builder\n\nUsage:\n  build_models.py -d <dataset> -a <algorithm> -f <feature function> -o <output>\n  build_models.py -h\n  build_models.py -v\n\nOptions:\n  -h              Show this screen.\n  -v              Show version.\n  -a --algorithm  Choose algorithm used\n  -d --dataset    Choose the dataset used (ned.train, esp.train)\n  -f --feature    Choose the feature function used\n  -o --output     File name of the outputted model')
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
        elif opt in ('-f', '--feature'):
            feature_function = arg
        elif opt in ("-o", "--output"):
            output = arg
        elif opt == "-v":
            print("Current version: 1.0.0")
            sys.exit()

    createmodel(dataset, feature_function, output, algorithm)

if __name__ == "__main__":
    main(sys.argv[1:])
