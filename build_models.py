import getopt
import pickle
import sys
from features import *
from nltk.corpus import conll2002 as conll
from custom_chunker import ConsecutiveNPChunker


def create_model(dataset, feature_function, output, algorithm):
    training = conll.chunked_sents(dataset)  # Get training dataset
    model = ConsecutiveNPChunker(functions[feature_function], training, algorithm, verbose=3)  # Creates the model
    pickle_model(model, output)  # Pickles the model


def pickle_model(model, output):
    pickle.dump(model, open(output, "wb"))  # Saves the model using pickle


def main(argv):
    # defining the main arguments
    algorithm = ''
    dataset = ''
    feature_function = ''
    output = ''
    try:
        opts, args = getopt.getopt(argv, "hd:a:f:o:v", ["dataset=", "algorithm=", "feature=", "output="]) # Try to get the options and their arguments using getopt
    except getopt.GetoptError:
        print(f'Invalid option! Usage: build_models.py -o <outfile>') # If it fails print invalid option and exit
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h": # Prints a help message
            print(
                'Model Builder\n\nUsage:\n  build_models.py -d <dataset> -a <algorithm> -f <feature function> -o <output>\n  build_models.py -h\n  build_models.py -v\n\nOptions:\n  -h              Show this screen.\n  -v              Show version.\n  -a --algorithm  Choose algorithm used\n  -d --dataset    Choose the dataset used (ned.train, esp.train)\n  -f --feature    Choose the feature function used\n  -o --output     File name of the outputted model')
            sys.exit()
        elif opt in ('-d', "--dataset"): # This options lets you choose between the spanish and dutch dataset
            if arg in ("ned.train", "esp.train"): # check if the dataset is ned.train or esp.train
                dataset = arg # set the dataset arg
            else:
                print("Dataset should be ned.train or esp.train")
                sys.exit(2)
        elif opt in ("-a", "--algorithm"): # Lets you choose the algorithm used
            if arg in ("NaiveBayes", "DecisionTree", "IIS", "GIS"):
                algorithm = arg # Set the algorithm arg
            else:
                print(
                    'Invalid algorithm! Choose from the following: \"NaiveBayes\", \"DecisionTree\", \"IIS\", and \"GIS\"')
                sys.exit(2)
        elif opt in ('-f', '--feature'): # Lets you choose the feature function
            feature_function = arg
        elif opt in ("-o", "--output"): # Lets you set the output file name
            output = arg
        elif opt == "-v": # Prints the version of the script
            print("Current version: 1.0.0")
            sys.exit()

    create_model(dataset, feature_function, output, algorithm) # Creates and pickles the model with the given arguments


if __name__ == "__main__":
    main(sys.argv[1:])
