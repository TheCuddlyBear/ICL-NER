import getopt
import sys


def main(argv):
    output_path = ""
    try:
        opts, args = getopt.getopt(argv, "ho:", ["output="])
    except getopt.GetoptError:
        print(f'Invalid option! Usage: build_models.py -o <outfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print('build_models.py -o <outfile>')
            sys.exit()
        elif opt in ("-o", "--output"):
            output_path = arg
    print(f'Output file is: {output_path}')


if __name__ == "__main__":
    main(sys.argv[1:])
