import os
import os.path as osp
from argparse import ArgumentParser
from utils import print_training_params

def parse_settings():
    EXP_CHOICES = ['awp','imagenet_pretraining']
    parser = ArgumentParser(description='PyTorch code for TAR: Test-time '
                            'Augmentation for Robustness')
    parser.add_argument('--experiment', type=str, default='trades', 
                        help='method on which to test TAR', choices=EXP_CHOICES)
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        help='Dataset to attack')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='name of directory for saving results')
    parser.add_argument('--seed', type=int, default=0, 
                        help='for deterministic behavior')
    parser.add_argument('--batch-size', type=int, default=250,
                        help='batch size')
    parser.add_argument('--chunks', type=int, default=10, 
                        help='num of chunks in which to break the dataset')
    parser.add_argument('--num-chunk', type=int, default=None, 
                        help='index of chunk to evaluate on')
    parser.add_argument('--eval-files', action='store_true', default=False,
		                help='evaluate based on files at '
                             'checkpoint/logs/results_chunk*of*_*to*.txt')
    parser.add_argument('--eps', type=float, default=0.031,
                        help='attack strength')
                        #anti adversary arguments
    parser.add_argument('--k', type=int, default=0, 
                        help='number of iterations for generating anti adversary')
    parser.add_argument('--alpha', type=float, default=0.15,
                        help='learning rate for generating the anti adversary')
    args = parser.parse_args()


    # Log path: verify existence of checkpoint dir, or create it
    if not osp.exists(args.checkpoint):
        os.makedirs(args.checkpoint, exist_ok=True)

    args.adv_dir = osp.join(args.checkpoint, 'advs')
    if not osp.exists(args.adv_dir):
        os.makedirs(args.adv_dir, exist_ok=True)

    args.logs_dir = osp.join(args.checkpoint, 'logs')
    if not osp.exists(args.logs_dir):
        os.makedirs(args.logs_dir, exist_ok=True)

    # txt file with all params
    chunk = 'all' if args.num_chunk is None else args.num_chunk
    args.info_log = osp.join(args.checkpoint, f'info_chunk_{chunk}.txt')
    print_training_params(args, args.info_log)

    # final results
    args.final_results = osp.join(args.checkpoint, f'results.txt')

    return args
