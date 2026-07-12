# peptide-gpcr-benchmark

A compact, runnable harness for **benchmarking a peptide–GPCR interaction dataset**
before you trust any model built on it. It ships with synthetic demo data so the
whole pipeline runs on a laptop in seconds, and it is built to be pointed at a
proprietary assay table by changing one file.

> The demo data in `data/` is **synthetic and illustrative** — not real assay
> measurements. It exists only to exercise the methodology. Swap in the real
> table (`peptide_id, peptide_seq, receptor_id, receptor_seq, label`) for a live
> benchmark.

## Why this exists

Peptide–receptor interaction datasets **leak**. If peptides for the same receptor
land in both the training and test folds, a flexible model memorises *"this
receptor binds a lot"* instead of learning interaction rules — and the reported
AUC looks great but collapses on any new receptor. The one number that actually
matters when you benchmark such a dataset is the **optimism gap** between a naive
random split and a leakage-safe, receptor-grouped split.

## What it does

```
                 ┌──────────────┐
 sequences  ───▶ │  featurize   │  physicochemical + AA composition
                 │  (or ESM-2)  │  per peptide & per receptor, concatenated
                 └──────┬───────┘
                        ▼
        ┌───────────────────────────────┐
        │ benchmark harness             │
        │  • naive StratifiedKFold      │  ← optimistic (receptor leakage)
        │  • leakage-safe GroupKFold    │  ← honest (unseen receptors)
        │  baselines: Prevalence,       │
        │  LogReg, RandomForest,        │
        │  GradientBoosting             │
        │  metrics: AUROC, AUPR, P@k    │
        └───────────────┬───────────────┘
                        ▼
              optimism gap report
```

## Run it

```bash
pip install -r requirements.txt
python main.py
```

Example output on the demo data (5-fold CV, mean ± SD):

| split | model | AUROC | AUPR |
|-------|-------|------:|-----:|
| naive random | RandomForest | **0.692** | 0.647 |
| leakage-safe grouped | RandomForest | **0.551** | 0.509 |
| leakage-safe grouped | LogReg | **0.633** | 0.608 |

**Optimism gap (RandomForest): 0.692 → 0.551 (Δ = 0.14).** The tree model was
memorising receptor identity; on unseen receptors it collapses toward chance,
while the linear model's generalisable rule holds. Report the grouped number,
never the random one.

## Use your own data

```bash
python -m src.benchmark --data your_assay_table.csv
```

CSV columns: `peptide_seq, receptor_seq, receptor_id, label` (a binary
interaction call; a continuous pKd/pEC50 can be thresholded or the harness
extended to regression).

## ESM-2 embeddings (optional)

The self-contained path uses physicochemical + composition features (no model
download). To benchmark against a protein language model instead:

```bash
pip install fair-esm torch
python -m src.benchmark --features esm
```

This mean-pools ESM-2 (35M) embeddings per sequence and concatenates the pair —
a drop-in stronger featurizer to quantify how much a PLM buys you over classical
descriptors on *your* data.

## Layout

```
main.py              end-to-end: build demo data + run benchmark
src/featurize.py     physicochemical + composition features; optional ESM-2 hook
src/make_dataset.py  synthetic illustrative peptide-GPCR dataset generator
src/benchmark.py     leakage-safe vs naive split, baselines, metrics, optimism gap
requirements.txt     numpy, pandas, scikit-learn (fair-esm/torch optional)
```

## Scope

This is a **methodology demonstrator**, not a production model. On a real
engagement it is extended with: receptor-family–aware splits, calibration and
decision-threshold analysis, negative-sampling / decoy strategy, cold-start
(new-peptide *and* new-receptor) evaluation, and docking- or structure-derived
features where structures are available.

---
Dr. Sandeep Grover — computational biology & ML for protein/receptor interaction.
