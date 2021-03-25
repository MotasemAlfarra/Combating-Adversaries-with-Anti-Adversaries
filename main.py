import sys
import torch
import random
import argparse
import numpy as np
import os.path as osp
import torch.backends.cudnn as cudnn

from utils.utils import (get_model, print_to_log, eval_chunk,
                         eval_files)

# For deterministic behavior
cudnn.benchmark = False
cudnn.deterministic = True


def set_seed(device, seed=111):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    

def main(args):
    # Setup
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(DEVICE, args.seed)

    # Model
    print('Computing anti adversary with {} iterations and {} learning rate'.format(args.k, args.alpha))
    model = get_model(args.experiment, args.k, args.alpha, args.dataset)

    # Data
    if args.num_chunk is None: # evaluate sequentially
        log_files = []
        for num_chunk in range(1, args.chunks+1):
            log_file = eval_chunk(model, args.dataset, args.batch_size, 
                                  args.chunks, num_chunk, DEVICE, args)
            log_files.append(log_file)

        eval_files(log_files, args.final_results)
    else: # evaluate a single chunk and exit
        log_file = eval_chunk(model, args.dataset, args.batch_size, args.chunks, 
                              args.num_chunk, DEVICE, args)
        sys.exit()

if __name__ == "__main__":
    from utils.opts import parse_settings
    args = parse_settings()
    if args.eval_files:
        from glob import glob
        log_files = glob(osp.join(args.logs_dir, 
                         'results_chunk*of*_*to*.txt'))
        eval_files(log_files, args.final_results)
        sys.exit()
    else:
        main(args)
