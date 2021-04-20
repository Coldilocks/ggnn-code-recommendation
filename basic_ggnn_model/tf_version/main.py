from docopt import docopt
from basic_ggnn_model.tf_version.model import DenseGGNNModel
import sys, traceback
import pdb

def main():
    args = docopt(__doc__)
    try:
        model = DenseGGNNModel(args)

        if args['--evaluate']:
            model.example_evaluation()
        else:
            model.train()
    except:
        typ, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    main()