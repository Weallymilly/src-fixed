from __future__ import annotations
"""
Utilities for data processing.
"""
import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import Sequence, Tuple, List
import logging

logger = logging.getLogger(__name__)

VALID_AAS = "MRHKDESTNQCUGPAVIFYWLO"

"""
File formatting note.
Data should be preprocessed as a sequence of comma-separated ints with
sequences \n separated
"""

# Lookup tables
aa_to_int = {
    'M':1,
    'R':2,
    'H':3,
    'K':4,
    'D':5,
    'E':6,
    'S':7,
    'T':8,
    'N':9,
    'Q':10, 'C':11,
    'U':12,
    'G':13,
    'P':14,
    'A':15,
    'V':16,
    'I':17,
    'F':18,
    'Y':19,
    'W':20,
    'L':21,
    'O':22, #Pyrrolysine
    'X':23, # Unknown
    'Z':23, # Glutamic acid or GLutamine
    'B':23, # Asparagine or aspartic acid
    'J':23, # Leucine or isoleucine
    'start':24,
    'stop':25,
    '-':26,
}

int_to_aa = {value:key for key, value in aa_to_int.items()}

def get_aa_to_int() -> dict:
    """
    Get the lookup table (for easy import)
    """
    return aa_to_int

def get_int_to_aa() -> dict:
    """
    Get the lookup table (for easy import)
    """
    return int_to_aa

def aa_seq_to_int(s: str) -> List[int]:
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return [24] + [aa_to_int[a] for a in s] + [25]

def int_seq_to_aa(s: Sequence[int]) -> str:
    """
    Return the int sequence as a list for a given string of amino acids
    """
    return "".join([int_to_aa[i] for i in s])

def nonpad_len(batch: np.ndarray) -> np.ndarray:
    nonzero = batch > 0
    lengths = np.sum(nonzero, axis=1)
    return lengths    

def format_seq(seq: str, stop: bool=False) -> List[int]:
    """
    Takes an amino acid sequence, returns a list of integers in the codex of the babbler.
    Here, the default is to strip the stop symbol (stop=False) which would have 
    otherwise been added to the end of the sequence. If you are trying to generate
    a rep, do not include the stop. It is probably best to ignore the stop if you are
    co-tuning the babbler and a top model as well.
    """
    if stop:
        int_seq = aa_seq_to_int(seq.strip())
    else:
        int_seq = aa_seq_to_int(seq.strip())[:-1]
    return int_seq

def format_batch_seqs(seqs: Sequence[str]) -> np.ndarray:
    maxlen = -1
    for s in seqs:
        if len(s) > maxlen:
            maxlen = len(s)
    formatted = []
    for seq in seqs:
        pad_len = maxlen - len(seq)
        padded = np.pad(format_seq(seq), (0, pad_len), 'constant', constant_values=0)
        formatted.append(padded)
    return np.stack(formatted)

def is_valid_seq(seq: str, max_len: int=2000) -> bool:
    """
    True if seq is valid for the babbler, False otherwise.
    """
    l = len(seq)
    if (l < max_len) and set(seq) <= set(VALID_AAS):
        return True
    else:
        return False

def seqs_to_onehot(seqs: Sequence[str]) -> np.ndarray:
    seqs_arr = format_batch_seqs(seqs)
    num_positions = seqs_arr.shape[1]
    X = (np.arange(24)[None, None, :] == seqs_arr[..., None]).astype(int)
    return X.reshape(seqs_arr.shape[0], num_positions*24)

def seqs_to_binary_onehot(seqs: Sequence[str], wt: str) -> np.ndarray:
    seqs = np.array([list(s.upper()) for s in seqs])
    wt_arr = np.array(list(wt.upper()))
    X = (seqs != wt_arr)
    return X.astype(int)

def dict2str(d: dict) -> str:
    return ';'.join([f'{k}={v}' for k, v in d.items()])

def seq2mutation(seq: str, model, return_str: bool=False, ignore_gaps: bool=False,
        sep: str=":", offset: int=1):
    mutations = []
    for pf, pm in model.index_map.items():
        if seq[pf-offset] != model.target_seq[pm]:
            if ignore_gaps and (
                    seq[pf-offset] == '-' or seq[pf-offset] not in model.alphabet):
                continue
            mutations.append((pf, model.target_seq[pm], seq[pf-offset]))
    if return_str:
        return sep.join([m[1] + str(m[0]) + m[2] for m in mutations])
    return mutations

def seq2mutation_fromwt(seq: str, wt: str, ignore_gaps: bool=False, sep: str=':', offset: int=1,
        focus_only: bool=True):
    mutations = []
    for i in range(offset, offset+len(seq)):
        if ignore_gaps and ( seq[i-offset] == '-'):
            continue
        if wt[i-offset].islower() and focus_only:
            continue
        if seq[i-offset].upper() != wt[i-offset].upper():
            mutations.append((i, wt[i-offset].upper(), seq[i-offset].upper()))
    return mutations

def seqs2subs(seqs: Sequence[str], wt: str, ignore_gaps: bool=False) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    pos = []
    subs = []
    for s in seqs:
        p = []
        su = []
        for j in range(len(wt)):
            if s[j] != wt[j]:
                if ignore_gaps and (s[j] == '-' or s[j] == 'X'):
                    continue
                p.append(j)
                su.append(s[j])
        pos.append(np.array(p))
        subs.append(np.array(su))
    return pos, subs

def seq2effect(seqs: Sequence[str], model, offset: int=1, ignore_gaps: bool=False) -> np.ndarray:
    effects = np.zeros(len(seqs))
    for i in range(len(seqs)):
        mutations = seq2mutation(seqs[i], model,
                ignore_gaps=ignore_gaps, offset=offset)
        dE, _, _ = model.delta_hamiltonian(mutations)
        effects[i] = dE
    return effects

def mutant2seq(mut: str, wt: str, offset: int) -> str:
    if mut.upper() == 'WT':
        return wt
    chars = list(wt)
    mut = mut.replace(':', ',')
    mut = mut.replace(';', ',')
    for m in mut.split(','):
        idx = int(m[1:-1])-offset
        assert wt[idx] == m[0]
        chars[idx] = m[-1]
    return ''.join(chars)

def get_blosum_scores(seqs: Sequence[str], wt: str, matrix) -> np.ndarray:
    scores = np.zeros(len(seqs))
    wt_score = 0
    for j in range(len(wt)):
        wt_score += matrix[wt[j], wt[j]]
    for i, s in enumerate(seqs):
        for j in range(len(wt)):
            if s[j] not in matrix.alphabet:
                logger.warning(f'unexpected AA {s[j]} (seq {i}, pos {j})')
            scores[i] += matrix[wt[j], s[j]]
    return scores - wt_score

def get_wt_seq(mutation_descriptions: Sequence[str]) -> Tuple[str, int]:
    wt_len = 0
    for m in mutation_descriptions:
        if m == 'WT':
            continue
        if int(m[1:-1]) > wt_len:
            wt_len = int(m[1:-1])
    wt = ['?' for _ in range(wt_len)]
    for m in mutation_descriptions:
        if m == 'WT':
            continue
        idx, wt_char = int(m[1:-1])-1, m[0]   # 1-index to 0-index
        if wt[idx] == '?':
            wt[idx] = wt_char
        else:
            assert wt[idx] == wt_char
    return ''.join(wt), wt_len
