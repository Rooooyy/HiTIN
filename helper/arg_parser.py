import argparse
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--config_file', type=str, required=True)

    # training
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('-l2', '--l2rate', type=float, default=0,
                        help='L2 penalty lambda (default: 0.01)')
    parser.add_argument('-p', '--load_pretrained', default=False, action='store_true')
    # TIN model
    parser.add_argument('-k', '--tree_depth', type=int, default=2,
                        help='The depth of coding tree to be constructed by CIRCA (default: 2)')
    parser.add_argument('-lm', '--num_mlp_layers', type=int, default=2,
                        help='Number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('-hd', '--hidden_dim', type=int, default=512,
                        help='Number of hidden units for HiTIN layer (default: 512)')
    parser.add_argument('-fd', '--final_dropout', type=float, default=0.5,
                        help='Dropout rate for HiTIN layer (default: 0.5)')
    parser.add_argument('-tp', '--tree_pooling_type', type=str, default="sum", choices=["root", "sum", "avg", "max"],
                        help='Pool strategy for the whole tree in Eq.11. Could be chosen from {root, sum, avg, max}.')
    # HTC
    parser.add_argument('-hp', '--hierar_penalty', type=float, default=0,
                        help='The weight for L^R in Eq.14 (default: 1e-6).')
    parser.add_argument('-ct', '--classification_threshold', type=float, default=0.5,
                        help='Threshold of binary classification. (default: 0.5)')
    # dirs
    parser.add_argument('--log_dir', type=str, default='log', help='Path to save log files (default: log).')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', help='Path to save checkpoints (default: ckpt).')

    parser.add_argument('--begin_time', type=str, default=time.strftime("%m%d_%H%M_", time.localtime()),
                        help='The beginning time of a run, which prefixes the name of log files.')

    args = parser.parse_args()
    return args
