"""
Feature extraction for peptide-GPCR interaction benchmarking.

Two self-contained featurizers (no external model downloads required) plus an
optional ESM-2 embedding hook that is used automatically when `fair-esm` +
torch are installed. The self-contained path lets the whole benchmark run on a
laptop with only numpy/pandas/scikit-learn.

Peptide features:
  - Amino-acid composition (20 dims)
  - Physicochemical descriptors (Kyte-Doolittle hydropathy, net charge at pH 7,
    aromaticity, mean molecular weight, isoelectric-ish charge fraction)
  - Sequence length (log-scaled)

Receptor features:
  - Amino-acid composition of the binding-relevant sequence (20 dims)
  - The same physicochemical descriptor block

An interaction example is the concatenation [peptide_feats | receptor_feats],
which is the standard "protein-protein / peptide-protein pair" encoding used in
interaction-prediction baselines.
"""
from __future__ import annotations

import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}

# Kyte-Doolittle hydropathy
KD = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
# Monoisotopic residue masses (Da)
MW = {
    "A": 71.04, "R": 156.10, "N": 114.04, "D": 115.03, "C": 103.01, "Q": 128.06,
    "E": 129.04, "G": 57.02, "H": 137.06, "I": 113.08, "L": 113.08, "K": 128.09,
    "M": 131.04, "F": 147.07, "P": 97.05, "S": 87.03, "T": 101.05, "W": 186.08,
    "Y": 163.06, "V": 99.07,
}
CHARGE_POS = set("KR")          # + at pH 7 (His treated as ~partial, ignored)
CHARGE_NEG = set("DE")          # - at pH 7
AROMATIC = set("FWY")


def _clean(seq: str) -> str:
    return "".join(c for c in seq.upper() if c in AA_IDX)


def aa_composition(seq: str) -> np.ndarray:
    seq = _clean(seq)
    v = np.zeros(20, dtype=np.float64)
    if not seq:
        return v
    for c in seq:
        v[AA_IDX[c]] += 1.0
    return v / len(seq)


def physchem(seq: str) -> np.ndarray:
    seq = _clean(seq)
    n = len(seq)
    if n == 0:
        return np.zeros(5, dtype=np.float64)
    hydropathy = np.mean([KD[c] for c in seq])
    net_charge = (sum(c in CHARGE_POS for c in seq) - sum(c in CHARGE_NEG for c in seq)) / n
    aromaticity = sum(c in AROMATIC for c in seq) / n
    mean_mw = np.mean([MW[c] for c in seq])
    charged_frac = sum(c in CHARGE_POS or c in CHARGE_NEG for c in seq) / n
    return np.array([hydropathy, net_charge, aromaticity, mean_mw, charged_frac])


def featurize_sequence(seq: str) -> np.ndarray:
    seq = _clean(seq)
    length = np.array([np.log1p(len(seq))])
    return np.concatenate([aa_composition(seq), physchem(seq), length])


def featurize_pairs(peptides, receptors) -> np.ndarray:
    """Return an (N, D) matrix for N peptide-receptor pairs."""
    rows = []
    for p, r in zip(peptides, receptors):
        rows.append(np.concatenate([featurize_sequence(p), featurize_sequence(r)]))
    return np.vstack(rows)


# ---- optional ESM-2 embeddings -------------------------------------------------
def esm_available() -> bool:
    try:
        import torch  # noqa: F401
        import esm  # noqa: F401
        return True
    except Exception:
        return False


def featurize_pairs_esm(peptides, receptors, model_name="esm2_t12_35M_UR50D"):
    """Mean-pooled ESM-2 embeddings concatenated for each pair.

    Used automatically by benchmark.py when fair-esm + torch are installed.
    Kept intentionally small (35M) so it runs without a GPU.
    """
    import torch
    import esm as esm_pkg

    model, alphabet = getattr(esm_pkg.pretrained, model_name)()
    bc = alphabet.get_batch_converter()
    model.eval()
    layer = model.num_layers

    def embed(seqs):
        out = []
        with torch.no_grad():
            for s in seqs:
                s = _clean(s)[:1022]
                _, _, toks = bc([("x", s)])
                rep = model(toks, repr_layers=[layer])["representations"][layer]
                out.append(rep[0, 1:len(s) + 1].mean(0).numpy())
        return np.vstack(out)

    return np.concatenate([embed(peptides), embed(receptors)], axis=1)
