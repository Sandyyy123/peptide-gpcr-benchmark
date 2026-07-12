"""
Generate a realistic *illustrative* peptide-GPCR interaction dataset.

IMPORTANT: this is SYNTHETIC demonstration data, not real assay measurements.
Its only purpose is to let the benchmarking harness run end-to-end so the
methodology (leakage-safe splits, baselines, metrics) can be inspected. On a
real engagement this loader is swapped for the client's proprietary assay table
(columns: peptide_id, peptide_seq, receptor_id, receptor_seq, label / pKd / pEC50).

The synthetic label has two ingredients, chosen so the benchmark reveals the
classic interaction-dataset pitfall:

  1. A GENERALISABLE peptide rule -- net charge and hydropathy of the peptide
     shift binding probability the same way for every receptor. This transfers
     to receptors never seen in training.

  2. A MEMORISABLE per-receptor base rate -- each receptor has its own binding
     prevalence (0.15-0.85), and each receptor is drawn from its own distinctive
     amino-acid distribution (Dirichlet), so its sequence features act as a
     fingerprint. A flexible model can memorise "this receptor binds a lot" when
     the receptor appears in BOTH train and test (random split), but that
     knowledge is useless for a brand-new receptor (grouped split).

Result: random-split AUROC is inflated by receptor memorisation, grouped-split
AUROC reflects honest generalisation, and the gap between them is the number
that actually matters when you benchmark a proprietary interaction dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AA = list("ACDEFGHIKLMNPQRSTVWY")
CHARGE_POS, CHARGE_NEG = set("KR"), set("DE")
KD = {  # Kyte-Doolittle hydropathy
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def _seq_from_probs(rng, probs, lo, hi):
    n = rng.integers(lo, hi + 1)
    return "".join(rng.choice(AA, size=n, p=probs))


def _net_charge(seq):
    n = max(len(seq), 1)
    return (sum(c in CHARGE_POS for c in seq) - sum(c in CHARGE_NEG for c in seq)) / n


def _hydropathy(seq):
    return np.mean([KD[c] for c in seq]) if seq else 0.0


def _logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return np.log(p / (1 - p))


def make(n_receptors=40, peptides_per_receptor=60, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_receptors):
        rid = f"GPCR_{i:03d}"
        # distinctive amino-acid distribution -> receptor features are a fingerprint
        rprobs = rng.dirichlet(np.full(20, 0.5))
        rseq = _seq_from_probs(rng, rprobs, 300, 360)
        base_rate = float(rng.uniform(0.15, 0.85))     # memorisable per-receptor prevalence
        base_logit = _logit(base_rate)
        for j in range(peptides_per_receptor):
            pprobs = rng.dirichlet(np.full(20, 1.0))
            pseq = _seq_from_probs(rng, pprobs, 8, 25)
            # generalisable peptide rule (shared across all receptors)
            g = 2.2 * _net_charge(pseq) + 0.35 * _hydropathy(pseq)
            logit = base_logit + g
            prob = 1.0 / (1.0 + np.exp(-logit))
            label = int(rng.random() < prob)
            rows.append({
                "peptide_id": f"{rid}_pep{j:03d}",
                "peptide_seq": pseq,
                "receptor_id": rid,
                "receptor_seq": rseq,
                "label": label,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    df = make()
    out = os.path.join(os.path.dirname(__file__), "..", "data", "peptide_gpcr_demo.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} pairs across {df.receptor_id.nunique()} receptors "
          f"to {os.path.normpath(out)} (positive rate {df.label.mean():.1%})")
