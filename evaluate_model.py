import pickle
import sys
import getopt
from nltk.corpus import conll2002 as conll

def main(argv):
    input = ''
    dataset = ''
    try:
        opts, args = getopt.getopt(argv, "hi:d:", ["input=", "dataset="])
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

    test = conll.chunked_sents(dataset)
    ner = pickle.load(open(input, "rb"))
    print(ner.accuracy(test))

if __name__ == "__main__":
    main(sys.argv[1:])