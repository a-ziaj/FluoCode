"""
v2: DMS-pretrained ESM2 + XGBoost, 5-fold cluster CV.

Stage 1  — pretrain ESM2 on Sarkisyan avGFP brightness data
Stage 2a — fine-tune on FPbase spectral labels (masked MSE)
Stage 2b — extract 480-dim embeddings from frozen encoder
Stage 2c — train one XGBoost regressor per target

Cluster-based CV keeps entire protein families in the same fold
to avoid leaking sequence similarity between train and test.
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluocode.data import load_fpbase, fetch_dms
from fluocode.model import ESM2Regressor, make_lora_config
from fluocode.train import pretrain_on_dms, finetune_on_fpbase, extract_embeddings
from fluocode.cv import cluster_sequences, get_cv_splits
from fluocode.evaluate import (fit_scalers, scale_targets, train_xgb_regressors,
                                aggregate_cv_results, print_cv_summary, TARGETS)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(out_dir) / f"v2_{RUN_ID}.log"),
        ],
    )


def run_fold(fold, train_idx, test_idx, seqs, tgt_scaled, scalers,
             tokenizer, device, cfg):
    log = logging.getLogger(__name__)
    log.info(f"\nfold {fold+1}/{cfg['n_folds']} — "
             f"{len(train_idx)} train / {len(test_idx)} test")

    tr_idx, val_idx = train_test_split(train_idx, test_size=0.15,
                                        random_state=cfg["seed"])

    dms_lora = make_lora_config(cfg["dms_lora_r"], cfg["dms_lora_alpha"], 0.05)
    model = ESM2Regressor(cfg["model_name"], dms_lora, output_dim=1).to(device)
    pretrain_on_dms(model, dms_df, tokenizer, device, cfg)

    enc_base    = {k: v.clone() for k, v in model.encoder.state_dict().items()
                   if "lora_A" not in k and "lora_B" not in k}
    fpbase_lora = make_lora_config(cfg["fpbase_lora_r"], cfg["fpbase_lora_alpha"],
                                    cfg["fpbase_lora_dropout"])
    model = ESM2Regressor(cfg["model_name"], fpbase_lora,
                           output_dim=len(TARGETS)).to(device)
    model.encoder.load_state_dict(enc_base, strict=False)
    finetune_on_fpbase(model, seqs, tgt_scaled, tr_idx, val_idx,
                       tokenizer, device, cfg)

    log.info("extracting embeddings...")
    X_tr  = extract_embeddings(model, seqs, tgt_scaled, tr_idx,   tokenizer, device, cfg)
    X_val = extract_embeddings(model, seqs, tgt_scaled, val_idx,  tokenizer, device, cfg)
    X_te  = extract_embeddings(model, seqs, tgt_scaled, test_idx, tokenizer, device, cfg)

    return train_xgb_regressors(
        X_tr, X_val, X_te,
        tgt_scaled[tr_idx], tgt_scaled[val_idx], tgt_scaled[test_idx],
        scalers, TARGETS, cfg["xgb_params"]
    )


def main(cfg):
    log = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device: {device}")
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        log.info(f"gpu: {p.name} ({p.total_memory/1e9:.0f}GB)")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    Path(cfg["results_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    global dms_df
    fpbase_df = load_fpbase(cfg["fpbase_cache"])
    dms_df    = fetch_dms(cfg["dms_cache"], cfg["dms_sample"])

    if cfg.get("test_run"):
        fpbase_df = fpbase_df.head(120).reset_index(drop=True)
        dms_df    = dms_df.head(400).reset_index(drop=True)
        cfg.update({"dms_epochs": 2, "fpbase_epochs": 3, "n_folds": 2})
        log.info("test run: reduced dataset and epochs")

    seqs       = fpbase_df["sequence"].tolist()
    scalers    = fit_scalers(fpbase_df[TARGETS])
    tgt_scaled = scale_targets(fpbase_df[TARGETS], scalers)
    tokenizer  = AutoTokenizer.from_pretrained(cfg["model_name"])

    fold_labels  = cluster_sequences(seqs, cfg.get("cluster_threshold", 0.3),
                                      cfg["n_folds"])
    splits       = get_cv_splits(fold_labels, cfg["n_folds"])
    fold_results = [run_fold(i, tr, te, seqs, tgt_scaled, scalers,
                             tokenizer, device, cfg)
                    for i, (tr, te) in enumerate(splits)]

    summary = aggregate_cv_results(fold_results, TARGETS)
    print_cv_summary(summary, TARGETS, "v2 (DMS + XGBoost)")

    out = {
        "model": "v2", "run_id": RUN_ID,
        "description": "DMS-pretrained ESM2 + XGBoost, cluster CV",
        "n_folds": cfg["n_folds"],
        "cv_summary": summary, "fold_results": fold_results,
        "config": {k: v for k, v in cfg.items() if not isinstance(v, dict)},
    }
    out_path = Path(cfg["results_dir"]) / f"v2_{RUN_ID}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"results: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/v2.yaml")
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["test_run"] = args.test_run

    setup_logging(cfg["results_dir"])
    logging.getLogger(__name__).info(f"config: {args.config} | run: {RUN_ID}")
    main(cfg)
