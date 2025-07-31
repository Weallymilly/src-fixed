import numpy as np
from typing import Dict, Tuple

# Simple parser for the .j file output from PLMC, which contains fields h_i(a)
# and couplings J_ij(a,b). Format is described in PLMC repo; adjust if your .j differs.

class CouplingsModel:
    def __init__(self, j_path: str):
        """
        j_path: path to the PLMC .j output file (e.g., model.j)
        """
        self.j_path = j_path
        self.h, self.J, self.alphabet, self.length = self._load_j_file(j_path)
        # Build index_map and target_seq placeholders if needed downstream
        # Here, assumption: target sequence is not in .j; user should supply or derive it.
        self.index_map = {i: i for i in range(self.length)}  # identity mapping
        self.target_seq = None  # Should be set externally if needed

    def _load_j_file(self, path: str) -> Tuple[np.ndarray, np.ndarray, str, int]:
        """
        Parse PLMC .j format for fields and couplings.
        Returns:
            h: (L, q) array of local fields (q = alphabet size)
            J: (L, L, q, q) array of couplings
            alphabet: string of symbols
            length: L
        """
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]

        # First pass: find alphabet line and length if present.
        # PLMC .j file has lines like:
        #   # alphabet: ACDEFGHIKLMNPQRSTVWY
        #   # length: 100
        alphabet = None
        length = None
        for l in lines:
            if l.startswith('#') and 'alphabet:' in l:
                alphabet = l.split('alphabet:')[1].strip()
            if l.startswith('#') and 'length:' in l:
                length = int(l.split('length:')[1].strip())
        if alphabet is None or length is None:
            raise ValueError("Could not parse alphabet/length from .j file header.")

        q = len(alphabet)
        L = length

        # Initialize h and J
        h = np.zeros((L, q))
        J = np.zeros((L, L, q, q))

        # Data lines encode parameters: format depends on plmc version
        # Example line for field: "1  A  <value>"  (position, symbol, field)
        # Example line for coupling: "1 2  A  C  <value>" (pos1, pos2, sym1, sym2, coupling)
        for l in lines:
            if l.startswith('#'):
                continue
            parts = l.split()
            if len(parts) == 3:
                # field term
                i = int(parts[0]) - 1  # plmc is 1-indexed
                a = parts[1]
                val = float(parts[2])
                ai = alphabet.index(a)
                h[i, ai] = val
            elif len(parts) == 5:
                i = int(parts[0]) - 1
                j = int(parts[1]) - 1
                a = parts[2]
                b = parts[3]
                val = float(parts[4])
                ai = alphabet.index(a)
                bi = alphabet.index(b)
                J[i, j, ai, bi] = val
                J[j, i, bi, ai] = val  # ensure symmetry if appropriate
            else:
                # ignore unrecognized lines
                continue

        return h, J, alphabet, L

    def to_independent_model(self):
        """
        Returns a model where couplings are removed (only fields remain).
        """
        indep = CouplingsModel.__new__(CouplingsModel)
        indep.h = self.h.copy()
        indep.J = np.zeros_like(self.J)
        indep.alphabet = self.alphabet
        indep.length = self.length
        indep.index_map = self.index_map
        indep.target_seq = self.target_seq
        indep.j_path = self.j_path
        return indep

    def energy(self, seq: str) -> float:
        """
        Compute Potts energy: sum_i h_i(a_i) + sum_{i<j} J_ij(a_i, a_j)
        """
        if len(seq) != self.length:
            raise ValueError(f"Sequence length {len(seq)} != model length {self.length}")
        energy = 0.0
        for i, ai in enumerate(seq):
            ai_idx = self.alphabet.index(ai)
            energy += self.h[i, ai_idx]
        for i in range(self.length):
            ai = seq[i]
            ai_idx = self.alphabet.index(ai)
            for j in range(i+1, self.length):
                aj = seq[j]
                aj_idx = self.alphabet.index(aj)
                energy += self.J[i, j, ai_idx, aj_idx]
        return energy

    def sequence_effect(self, seq: str, wt: str) -> float:
        """
        Effect relative to wild-type: difference in energy.
        """
        return self.energy(seq) - self.energy(wt)
