from __future__ import annotations
'''
Script modified from https://github.com/chloechsu/combining-evolutionary-and-assay-labelled-data
python src/evaluate.py phytase onehot --n_seeds=1 --n_threads=1 --n_train=-1
'''
import logging
import os

import numpy as np
import pandas as pd

from predictors import get_predictor_cls, BoostingPredictor, JointPredictor
from utils.metric_utils import spearman, topk_mean, r2, hit_rate, aucroc, ndcg
from utils.io_utils import load_data_split, get_wt_log_fitness, get_log_fitness_cutoff
from utils.data_utils import dict2str

logger = logging.getLogger(__name__)

MAX_N_TEST=10000


def evaluate_predictor(dataset_name: str, predictor_name: str, joint_training: bool,
        n_train: int, metric_topk, max_n_mut: int, train_on_single, ignore_gaps: bool,
        seed: int, predictor_params: dict, outpath: str) -> None:
    print(f'----- predictor {predictor_name}, seed {seed} -----')
    outpath = f'{outpath}-{os.getpid()}'  # each process writes to its own file
    data = load_data_split(dataset_name, split_id=-1,
            ignore_gaps=ignore_gaps)

    predictor_cls = get_predictor_cls(predictor_name)
    if len(predictor_cls) == 1:
        predictor = predictor_cls[0](dataset_name, **predictor_params)
    elif joint_training:
        predictor = JointPredictor(dataset_name, predictor_cls,
            predictor_name, **predictor_params)
    else:
        predictor = BoostingPredictor(dataset_name, predictor_cls,
            **predictor_params)

    #test = data.sample(frac=0.2, random_state=seed)
    test = data[350:] #line number - 1
    print(test)
    # if len(test) > MAX_N_TEST:
    #     test = test.sample(n=MAX_N_TEST, random_state=seed)
    test = test.copy()
    train = data.drop(test.index)
    print(train)
    print(len(test), len(train))
    assert len(train) >= n_train, 'not enough training data'

    if n_train == -1:
        n_train = len(train)
        predictor.train(train.seq.values, train.log_fitness.values)
        test['pred'] = predictor.predict(test.seq.values)
        print(len(test['pred']), 'saving csv')
        csv_dir = 'results/phytase/round3/'
        os.makedirs(csv_dir, exist_ok=True)
        csv_path = f'{csv_dir}predicted-value-round3-quad-{seed}.csv'
        test.to_csv(csv_path, index=False, columns = ['mutant', 'pred'])
        logger.info('csv saved')

def run_from_queue(worker_id: int, queue) -> None:
    while True:
        args = queue.get()
        try:
            evaluate_predictor(*args)
        except Exception as e:
            logging.error("ERROR: %s", str(e))
            logging.exception(e)
            queue.task_done()
        queue.task_done()