from __future__ import annotations
'''
Script obtained from https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data

'''
import argparse
from multiprocessing import Process, JoinableQueue
import os

import pandas as pd

from utils import parse_vars, merge_dfs
from modified_eval import run_from_queue
from predictors import get_predictor_names 


def main():
    parser = argparse.ArgumentParser(
            description='Example: python evaluate.py sarkisyan onehot_ridge '
            '--predictor_params reg_coef=0.01')
    parser.add_argument('dataset_name', type=str,
            help='Dataset name. Folder of the same name under the data '
            'and inference directories are expected to look up files.'
            'The data will be loaded from data/{dataset_name}/data.csv'
            'in the `seq` and `log_fitness` columns.')
    parser.add_argument('predictor_name', type=str,
            help='Predictor name, or all for running all predictors.')
    parser.add_argument('--n_threads', type=int, default=20)
    parser.add_argument('--n_train', type=int, default=96)
    parser.add_argument('--max_n_mut', type=int, default=5)
    parser.add_argument('--joint_training', dest='joint_training', action='store_true')
    parser.add_argument('--boosting', dest='joint_training', action='store_false')
    parser.set_defaults(joint_training=True)
    parser.add_argument('--train_on_single', dest='train_on_single', action='store_true')
    parser.add_argument('--train_on_all', dest='train_on_single', action='store_false')
    parser.set_defaults(train_on_single=True)
    parser.add_argument('--ignore_gaps', dest='ignore_gaps', action='store_true')
    parser.set_defaults(ignore_gaps=False)
    parser.add_argument('--n_seeds', type=int, default=20,
            help='Number of random train test splits to get confidence interval')
    parser.add_argument('--metric_topk', type=int, default=96,
            help='Top ? when evaluating hit rate and topk mean')
    parser.add_argument("--predictor_params",
                        metavar="KEY=VALUE",
                        nargs='+',
                        help="Set a number of key-value pairs "
                             "(do not put spaces before or after the = sign). "
                             "If a value contains spaces, you should define "
                             "it with double quotes: "
                             'foo="this is a sentence". Note that '
                             "values are always treated as floats.")
    parser.add_argument('--results_suffix', type=str, default='')
    args = parser.parse_args()
    predictor_params = parse_vars(args.predictor_params or [])
    if args.ignore_gaps:
        predictor_params['ignore_gaps'] = args.ignore_gaps
    print(args)

    outdir = os.path.join('results', args.dataset_name)
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f'results{args.results_suffix}.csv')

    # Multiprocessing setup (note: on macOS the default start method may need 'spawn', but keeping as-is)
    queue = JoinableQueue()
    workers = []
    for i in range(args.n_threads):
        p = Process(target=run_from_queue, args=(i, queue))
        workers.append(p)
        p.start()
    predictors = get_predictor_names(args.predictor_name)
    for pn in predictors:
        for seed in range(args.n_seeds):
            queue.put((args.dataset_name, pn, args.joint_training,
                args.n_train, args.metric_topk, args.max_n_mut,
                args.train_on_single, args.ignore_gaps, seed,
                predictor_params, outpath))
    queue.join()
    for p in workers:
        p.terminate()
    merge_dfs(f'{outpath}*', outpath,
            index_cols=['dataset', 'predictor', 'predictor_params', 'seed'],
            groupby_cols=['predictor', 'predictor_params', 'n_train'],
            ignore_cols=['seed'])


if __name__ == '__main__':
    print(pd.__version__)
    main()
