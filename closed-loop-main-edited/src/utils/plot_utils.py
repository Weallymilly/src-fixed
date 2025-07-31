from __future__ import annotations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from typing import Sequence, Callable
from utils.metric_utils import spearman, topk_mean, hit_rate, aucroc


def get_stratified_metrics(df: pd.DataFrame, model_name: str, max_n_mut: int, metric_fn: Callable[[np.ndarray, np.ndarray], float]) -> np.ndarray:
    """Calculate stratified metrics for each mutation count from 0 to max_n_mut."""
    strat_metrics = np.zeros(max_n_mut+1)
    for i in range(0, max_n_mut+1):
        # 0th position is overall aggregated metric
        tmp = df
        if i > 0:
            tmp = df[df.n_mut == i]
        strat_metrics[i] = metric_fn(tmp[model_name], tmp.log_fitness)
    return strat_metrics


def plot_stratified_metrics(ax: plt.Axes, df: pd.DataFrame, models: Sequence[str], max_n_mut: int, metric_fn: Callable[[np.ndarray, np.ndarray], float], vmin: float|None, vmax: float|None) -> None:
    """Plot heatmap of stratified metrics for given models and mutation counts."""
    strat_matrix = np.zeros((len(models), 1+max_n_mut))
    xticklabels=['All'] + list(range(1, max_n_mut+1))
    for i, m in enumerate(models):
        strat_matrix[i] = get_stratified_metrics(df, m, max_n_mut, metric_fn)    
    sns.heatmap(strat_matrix, yticklabels=models, xticklabels=xticklabels,
                vmin=vmin, vmax=vmax, ax=ax, cmap='viridis')
    ax.set_xlabel('# Mutations')
    ax.vlines([1], *ax.get_ylim(), colors='black')
    

def plot_auc_and_corr(df: pd.DataFrame, models: Sequence[str], functional_threshold: float, wt_log_fitness: float,
        max_n_mut: int=5, vmin=None, vmax=None, topk: int=96) -> None:
    """Plot AUC-ROC and correlation metrics stratified by mutation counts for given models."""
    vmin = [None, None, None] if vmin is None else vmin
    vmax = [None, None, None] if vmax is None else vmax
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)

    ax = axes[0]
    fn = partial(aucroc, y_cutoff=functional_threshold)
    plot_stratified_metrics(ax, df, models, max_n_mut, fn, vmin[0], vmax[0])
    ax.set_title(f'Functional vs Non-Functional AUC-ROC')

    ax = axes[1]
    fn = partial(aucroc, y_cutoff=wt_log_fitness)
    plot_stratified_metrics(ax, df[df.log_fitness >= functional_threshold],
            models, max_n_mut, fn, vmin[1], vmax[1])
    ax.set_title(f'Functional, <WT vs >=WT AUC-ROC')
    
    ax = axes[2]
    plot_stratified_metrics(ax, df[df.log_fitness >= functional_threshold],
            models, max_n_mut, spearman, vmin[2], vmax[2])
    ax.set_title('Rank Correlation (Functional)')

    fig.suptitle('Model performance, stratified by # mutations')
    plt.subplots_adjust(wspace=0.1, top=0.85)
    plt.show()
