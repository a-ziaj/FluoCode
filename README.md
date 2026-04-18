# FluoCode

Predicting fluorescent protein spectral properties (λex, λem, extinction coefficient, quantum yield) from sequence and structure.

Two models:

- **v2** — ESM2 + LoRA fine-tuned on FPbase, XGBoost regression heads. Sequence only.
- **v6** — v2 + pairwise Cα distances from the chromophore pocket of energy-minimised structures.

Both evaluated with cluster-based 5-fold cross-validation.

---

## Why cluster-based CV

Random k-fold splits proteins arbitrarily. Since FPbase contains many closely related variants (mEGFP, EGFP, eGFP-S65T, ...), a random split puts these in both train and test — the model effectively memorises the family and the evaluation overstates generalisation.

Cluster CV groups proteins by sequence identity first, then assigns entire clusters to folds. The test set at each split is genuinely out-of-distribution. It's a harder and more honest evaluation.

---

## Pipeline
 
```
Sarkisyan DMS (51k avGFP variants)
  → ESM2 + LoRA (r=4)      pretrain on brightness regression
  → ESM2 + LoRA (r=8)      fine-tune on FPbase spectral labels (masked MSE)
  → 480-dim embedding       mean-pool over residues, freeze encoder
 
  v2  →  XGBoost × 4
  v6  →  concatenate 900-dim flat pocket distances  →  XGBoost × 4
  v7  →  GATv2 over pocket graph (ESM2 node features, Cα edge distances)
          → 128-dim pocket embedding
          → concatenate with 480-dim protein embedding
          → XGBoost × 4
```
 
The DMS pretraining step is important. ESM2 out of the box has no particular knowledge of what makes a fluorescent protein work. Training it to predict GFP variant brightness first gives it a useful prior — it learns which positions are sensitive before seeing any wavelength data.
 
The v7 GNN uses ESM2 per-residue embeddings as node features, so each pocket residue carries both its sequence context and its 3D neighbourhood. This is richer than v6's flat distance matrix, where the identity of each residue is lost.

---

## Results (fixed split, no CV)

| Property | v2 Spearman | v6 Spearman | v2 R² | v6 R² |
|----------|-------------|-------------|-------|-------|
| ex_max   | 0.758       | 0.752       | 0.541 | 0.551 |
| em_max   | 0.685       | **0.711**   | 0.567 | **0.592** |
| ext_coeff| **0.682**   | 0.681       | 0.472 | 0.497 |
| qy       | 0.638       | **0.645**   | **0.392** | 0.368 |

v6 gains most on em_max, which makes sense — emission wavelength is most directly determined by the electrostatic environment of the chromophore, so real 3D distances add signal that sequence alone can only infer indirectly.

---

## Installation

```bash
git clone https://github.com/yourusername/FluoCode
cd FluoCode
pip install -r requirements.txt
```

PyTorch install depends on your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/).

### env 

```bash
conda create -y -p fluocode python=3.10
conda activate fluocode
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Usage

```bash
# test run first (small data, 2 epochs, 2 folds)
python scripts/train_v2.py --config configs/v2.yaml --test_run
python scripts/train_v6.py --config configs/v6.yaml --test_run

# full run
python scripts/train_v2.py --config configs/v2.yaml
python scripts/train_v6.py --config configs/v6.yaml

# submit scripts
qsub run_v2.sh
qsub run_v6.sh
```

---

## Data

**FPbase** — downloaded automatically via REST API on first run. ~990 fluorescent proteins, manually curated from primary literature.
Lambert et al., *Nature Methods* 2019.

**Sarkisyan DMS** — downloaded from HuggingFace (`SaProtHub/Dataset-Fluorescence-TAPE`). 51,715 avGFP variants with log-fluorescence scores.
Sarkisyan et al., *Nature* 2016.

**Minimised structures (v6)** — place energy-minimised PDB files in `data/structure/minimized/` as `<protein_name>_minimized.pdb`. The name must match the FPbase entry (lowercased). Unmatched proteins use zero pocket features.

---

## Structure

```
fluocode/
    data.py       FPbase and DMS loading
    model.py      ESM2 + LoRA
    structure.py  PDB parsing and pocket feature extraction
    train.py      DMS pretraining and FPbase fine-tuning
    cv.py         cluster-based cross-validation
    evaluate.py   XGBoost, metrics, CV aggregation
scripts/
    train_v2.py   sequence-only model
    train_v6.py   sequence + structure model
configs/
    v2.yaml
    v6.yaml
```

---

## References

- Lin et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*.
- Lambert et al. (2019). FPbase: a community-editable fluorescent protein database. *Nature Methods*.
- Sarkisyan et al. (2016). Local fitness landscape of the green fluorescent protein. *Nature*.
- Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. *KDD*.
- Hu et al. (2021). LoRA: Low-rank adaptation of large language models. *ICLR*.
