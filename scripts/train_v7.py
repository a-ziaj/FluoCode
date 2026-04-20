"""
v7: DMS-pretrained ESM2 + pocket GNN + XGBoost

Same pipeline as v6, with one addition between embedding extraction
and XGBoost: a graph attention network runs over each protein's
chromophore pocket and produces a 128-dim pocket embedding.

Feature vector fed to XGBoost:
    480-dim ESM2 protein embedding     (sequence context, global)
  + 128-dim GNN pocket embedding       (local 3D geometry, learned)
  = 608-dim total

The GNN uses ESM2 per-residue embeddings as node features, so it
has access to both the sequence identity of each pocket residue and
its position in 3D space — neither v2 (sequence only) nor v6 (flat
distances) has both simultaneously.

Usage:
    python scripts/train_v7.py --config configs/v7.yaml
    python scripts/train_v7.py --config configs/v7.yaml --test_run
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
from torch_geometric.data import Batch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fluocode.data import load_fpbase, fetch_dms
from fluocode.model import ESM2Regressor, make_lora_config
from fluocode.structure import parse_ca_coords, build_pocket_features
from fluocode.gnn import PocketGNN, build_pocket_graph, extract_residue_embeddings
from fluocode.train import pretrain_on_dms, finetune_on_fpbase, extract_embeddings
from fluocode.cv import cluster_sequences, get_cv_splits
from fluocode.evaluate import (
    fit_scalers, scale_targets, train_xgb_regressors,
    aggregate_cv_results, print_cv_summary, TARGETS
)

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(out_dir) / f"v7_{RUN_ID}.log"),
        ],
    )


def build_graphs(fpbase_df, residue_embs, ca_coords_list, cfg):
    """
    Build one PyG graph per protein.
    Returns a list of Data objects (or None for proteins with no PDB).
    """
    log  = logging.getLogger(__name__)
    graphs, failed = [], 0

    for i, row in fpbase_df.iterrows():
        ca = ca_coords_list[i]
        if ca is None:
            graphs.append(None)
            failed += 1
            continue

        g = build_pocket_graph(
            ca_coords    = ca,
            residue_embs = residue_embs[i],
            chrom_res    = cfg.get("chromophore_res", 66),
            pocket_radius= cfg.get("pocket_radius_a", 8.0),
            edge_cutoff  = cfg.get("edge_cutoff_a", 6.0),
        )
        graphs.append(g)
        if g is None:
            failed += 1

    log.info(f"built {len(graphs) - failed}/{len(graphs)} pocket graphs")
    return graphs


def load_ca_coords(fpbase_df, cfg):
    """Load Cα coordinates from PDB files. Returns list of tensors or None."""
    from pathlib import Path as P
    import re

    sdir = P(cfg["structure_dir"])

    def variants(name):
        base = name.lower().strip()
        return list(dict.fromkeys([
            base, base.replace(" ", "-"), base.replace(" ", "_"),
            base.replace(" ", ""), re.sub(r"[^a-z0-9\-]", "", base),
            re.sub(r"[^a-z0-9]", "", base),
        ]))

    lookup = {}
    for p in sdir.glob("*_minimized.pdb"):
        stem = p.stem.replace("_minimized", "")
        lookup[stem] = p
        lookup[re.sub(r"[^a-z0-9]", "", stem.lower())] = p

    result = []
    for _, row in fpbase_df.iterrows():
        pdb = next((lookup[v] for v in variants(str(row.get("name", "")))
                    if v in lookup), None)
        if pdb is None:
            result.append(None)
            continue
        try:
            ca = torch.tensor(parse_ca_coords(pdb))
            result.append(ca)
        except Exception:
            result.append(None)
    return result


def gnn_embeddings(gnn, graphs, idx_list, device):
    """
    Run GNN over a subset of graphs and return embeddings.
    Proteins without a graph get zero vectors.
    """
    out_dim = gnn.conv3.out_channels
    result  = torch.zeros(len(idx_list), out_dim)

    valid_pos, valid_graphs = [], []
    for pos, i in enumerate(idx_list):
        g = graphs[i]
        if g is not None:
            valid_pos.append(pos)
            valid_graphs.append(g)

    if valid_graphs:
        batch = Batch.from_data_list(valid_graphs).to(device)
        gnn.eval()
        with torch.no_grad():
            embs = gnn(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        for pos, emb in zip(valid_pos, embs):
            result[pos] = emb.cpu()

    return result.numpy()


def run_fold(fold, train_idx, test_idx, seqs, tgt_scaled, scalers,
             fpbase_df, ca_coords_list, tokenizer, device, cfg):
    log = logging.getLogger(__name__)
    log.info(f"\nfold {fold+1}/{cfg['n_folds']} — "
             f"{len(train_idx)} train / {len(test_idx)} test")

    tr_idx, val_idx = train_test_split(train_idx, test_size=0.15,
                                        random_state=cfg["seed"])

    # Stage 1 — DMS pretraining
    dms_lora = make_lora_config(cfg["dms_lora_r"], cfg["dms_lora_alpha"], 0.05)
    model    = ESM2Regressor(cfg["model_name"], dms_lora, output_dim=1).to(device)
    pretrain_on_dms(model, dms_df, tokenizer, device, cfg)

    # Stage 2a — FPbase fine-tuning
    enc_base    = {k: v.clone() for k, v in model.encoder.state_dict().items()
                   if "lora_A" not in k and "lora_B" not in k}
    fpbase_lora = make_lora_config(cfg["fpbase_lora_r"], cfg["fpbase_lora_alpha"],
                                    cfg["fpbase_lora_dropout"])
    model = ESM2Regressor(cfg["model_name"], fpbase_lora,
                           output_dim=len(TARGETS)).to(device)
    model.encoder.load_state_dict(enc_base, strict=False)
    finetune_on_fpbase(model, seqs, tgt_scaled, tr_idx, val_idx,
                       tokenizer, device, cfg)

    # Stage 2b — ESM2 protein-level embeddings (480-dim)
    log.info("extracting protein embeddings...")
    X_tr_seq  = extract_embeddings(model, seqs, tgt_scaled, tr_idx,   tokenizer, device, cfg)
    X_val_seq = extract_embeddings(model, seqs, tgt_scaled, val_idx,  tokenizer, device, cfg)
    X_te_seq  = extract_embeddings(model, seqs, tgt_scaled, test_idx, tokenizer, device, cfg)

    # Stage 2c — ESM2 per-residue embeddings for GNN node features
    log.info("extracting per-residue embeddings for GNN...")
    all_seqs   = fpbase_df["sequence"].tolist()
    subset_idx = list(set(tr_idx.tolist() + val_idx.tolist() + test_idx.tolist()))
    subset_seqs = [all_seqs[i] for i in subset_idx]
    res_embs_list = extract_residue_embeddings(model, subset_seqs, tokenizer, device, cfg)
    # map back to full index space
    res_embs = {i: res_embs_list[pos] for pos, i in enumerate(subset_idx)}

    # Build pocket graphs using per-residue embeddings + 3D coordinates
    log.info("building pocket graphs...")
    graphs = []
    for i in range(len(fpbase_df)):
        ca = ca_coords_list[i]
        re = res_embs.get(i)
        if ca is None or re is None:
            graphs.append(None)
            continue
        # align lengths — ESM2 may truncate very long sequences
        min_len = min(len(ca), len(re))
        g = build_pocket_graph(
            ca_coords    = ca[:min_len],
            residue_embs = re[:min_len],
            chrom_res    = cfg.get("chromophore_res", 66),
            pocket_radius= cfg.get("pocket_radius_a", 8.0),
            edge_cutoff  = cfg.get("edge_cutoff_a", 6.0),
        )
        graphs.append(g)

    n_graphs = sum(1 for g in graphs if g is not None)
    log.info(f"  {n_graphs}/{len(graphs)} pocket graphs built")

    # Stage 2d — train GNN on training fold
    log.info("training pocket GNN...")
    gnn = train_gnn(gnn_model=PocketGNN(
                        node_dim=model.encoder.config.hidden_size,
                        hidden=cfg.get("gnn_hidden", 256),
                        out_dim=cfg.get("gnn_out_dim", 128),
                        heads=cfg.get("gnn_heads", 4),
                        dropout=cfg.get("gnn_dropout", 0.2)),
                    graphs=graphs,
                    idx_tr=tr_idx, idx_val=val_idx,
                    tgt_scaled=tgt_scaled,
                    cfg=cfg, device=device)

    # Stage 2e — extract GNN pocket embeddings
    X_tr_gnn  = gnn_embeddings(gnn, graphs, tr_idx,   device)
    X_val_gnn = gnn_embeddings(gnn, graphs, val_idx,  device)
    X_te_gnn  = gnn_embeddings(gnn, graphs, test_idx, device)

    # Concatenate: [ESM2 | GNN pocket]
    X_tr  = np.hstack([X_tr_seq,  X_tr_gnn])
    X_val = np.hstack([X_val_seq, X_val_gnn])
    X_te  = np.hstack([X_te_seq,  X_te_gnn])
    log.info(f"combined features: {X_tr.shape[1]}-dim "
             f"({X_tr_seq.shape[1]} ESM2 + {X_tr_gnn.shape[1]} GNN)")

    return train_xgb_regressors(
        X_tr, X_val, X_te,
        tgt_scaled[tr_idx], tgt_scaled[val_idx], tgt_scaled[test_idx],
        scalers, TARGETS, cfg["xgb_params"]
    )


def train_gnn(gnn_model, graphs, idx_tr, idx_val, tgt_scaled, cfg, device):
    """
    Train the GNN to predict spectral targets from pocket graphs.
    Uses masked MSE so partial labels are handled correctly.
    Only proteins with a graph contribute to the loss.
    """
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from fluocode.train import masked_mse

    gnn    = gnn_model.to(device)
    out_dim_gnn = cfg.get("gnn_out_dim", 128)
    n_targets   = len(TARGETS)

    head = nn.Sequential(
        nn.Linear(out_dim_gnn, 64), nn.GELU(), nn.Linear(64, n_targets)
    ).to(device)

    opt   = AdamW(list(gnn.parameters()) + list(head.parameters()),
                  lr=cfg.get("gnn_lr", 1e-3), weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=cfg.get("gnn_epochs", 30), eta_min=1e-5)

    epochs    = cfg.get("gnn_epochs", 30)
    patience  = cfg.get("gnn_patience", 8)
    best_val, no_improve, best_state = float("inf"), 0, None

    for epoch in range(1, epochs + 1):
        gnn.train(); head.train()
        tr_loss, n_tr = 0.0, 0
        for i in idx_tr:
            g = graphs[i]
            if g is None:
                continue
            g   = g.to(device)
            emb = gnn(g.x, g.edge_index, g.edge_attr,
                      torch.zeros(g.num_nodes, dtype=torch.long, device=device))
            pred = head(emb)
            lbl  = torch.tensor(tgt_scaled[i], dtype=torch.float32,
                                device=device).unsqueeze(0)
            loss = masked_mse(pred, lbl)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(gnn.parameters()) + list(head.parameters()), 1.0)
            opt.step()
            tr_loss += loss.item(); n_tr += 1

        gnn.eval(); head.eval()
        vl_loss, n_val = 0.0, 0
        with torch.no_grad():
            for i in idx_val:
                g = graphs[i]
                if g is None:
                    continue
                g   = g.to(device)
                emb = gnn(g.x, g.edge_index, g.edge_attr,
                          torch.zeros(g.num_nodes, dtype=torch.long, device=device))
                pred = head(emb)
                lbl  = torch.tensor(tgt_scaled[i], dtype=torch.float32,
                                    device=device).unsqueeze(0)
                vl_loss += masked_mse(pred, lbl).item(); n_val += 1

        sched.step()
        if n_tr and n_val:
            tr_loss /= n_tr; vl_loss /= n_val
            if epoch % 5 == 0 or epoch == 1:
                logging.getLogger(__name__).info(
                    f"  GNN epoch {epoch:03d} | train {tr_loss:.4f} | val {vl_loss:.4f}")

            if vl_loss < best_val:
                best_val, no_improve = vl_loss, 0
                best_state = {k: v.cpu().clone()
                              for k, v in gnn.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if best_state:
        gnn.load_state_dict(best_state)
    return gnn


import torch.nn as nn   # needed inside train_gnn


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
        cfg.update({"dms_epochs": 2, "fpbase_epochs": 3,
                    "gnn_epochs": 5, "n_folds": 2})
        log.info("test run: reduced dataset and epochs")

    seqs       = fpbase_df["sequence"].tolist()
    scalers    = fit_scalers(fpbase_df[TARGETS])
    tgt_scaled = scale_targets(fpbase_df[TARGETS], scalers)
    tokenizer  = AutoTokenizer.from_pretrained(cfg["model_name"])

    log.info("loading PDB structures...")
    ca_coords_list = load_ca_coords(fpbase_df, cfg)
    n_pdb = sum(1 for c in ca_coords_list if c is not None)
    log.info(f"  {n_pdb}/{len(fpbase_df)} proteins have PDB structures")

    fold_labels  = cluster_sequences(seqs, cfg.get("cluster_threshold", 0.3),
                                      cfg["n_folds"])
    splits       = get_cv_splits(fold_labels, cfg["n_folds"])
    fold_results = [run_fold(i, tr, te, seqs, tgt_scaled, scalers,
                             fpbase_df, ca_coords_list, tokenizer, device, cfg)
                    for i, (tr, te) in enumerate(splits)]

    summary = aggregate_cv_results(fold_results, TARGETS)
    print_cv_summary(summary, TARGETS, "v7 (DMS + GNN pocket + XGBoost)")

    out = {
        "model": "v7", "run_id": RUN_ID,
        "description": "DMS-pretrained ESM2 + GATv2 pocket GNN + XGBoost, cluster CV",
        "n_folds": cfg["n_folds"],
        "cv_summary": summary, "fold_results": fold_results,
        "config": {k: v for k, v in cfg.items() if not isinstance(v, dict)},
    }
    out_path = Path(cfg["results_dir"]) / f"v7_{RUN_ID}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"results: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/v7.yaml")
    parser.add_argument("--test_run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["test_run"] = args.test_run

    setup_logging(cfg["results_dir"])
    logging.getLogger(__name__).info(f"config: {args.config} | run: {RUN_ID}")
    main(cfg)
