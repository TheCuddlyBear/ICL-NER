import pickle
import sys
import getopt
from nltk.corpus import conll2002 as conll
from custom_chunker import ConsecutiveNPChunker

def main(argv):
    input = ''
    dataset = ''
    output = None
    try:
        opts, args = getopt.getopt(argv, "hi:d:o:", ["input=", "dataset=", "output="])
    except getopt.GetoptError:
        print(f'Invalid option! Usage: evaluate_models.py -i <input file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("Usage: evaluate_models.py -i <input file>")
            sys.exit()
        elif opt in ("-i", "--input"):
            input = arg
        elif opt in ("-d", "--dataset"):
            if arg in ('ned.testa', 'ned.testb', 'esp.testa', 'esp.testb'):
                dataset = arg
            else:
                print("Dataset should be ned.testa/b or esp.testa/b")
                sys.exit(2)
        elif opt in ("-o", "--output"):
            output = arg


    test = conll.chunked_sents(dataset)
    ner: ConsecutiveNPChunker = pickle.load(open(input, "rb"))
    ner.explain()
    acc = ner.accuracy(test)
    print(acc)

    if output != None:
        with open(output, 'a') as file:
            file.write(f"Algorithm: {ner.tagger.algorithm}\n{ner.tagger.feature_function.__doc__}{acc}\n\n")


if __name__ == "__main__":
    main(sys.argv[1:])