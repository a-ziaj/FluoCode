import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from fluocode.data import SequenceDataset

log = logging.getLogger(__name__)


def masked_mse(pred, target):
    """MSE that ignores NaN entries — handles proteins with partial labels."""
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=pred.device)
    return nn.functional.mse_loss(pred[mask], target[mask])


def _make_loader(seqs, labels, tokenizer, max_length, batch_size, shuffle):
    ds = SequenceDataset(seqs, labels, tokenizer, max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=2, pin_memory=True)


def pretrain_on_dms(model, dms_df, tokenizer, device, cfg):
    """
    Stage 1: train on Sarkisyan brightness data before seeing any spectral labels.
    Simple MSE regression on log_fluorescence. Gives the encoder a functional
    prior — it learns what kills vs preserves GFP fluorescence.
    """
    log.info(f"--- DMS pretraining: {len(dms_df)} sequences, {cfg['dms_epochs']} epochs ---")

    seqs   = dms_df["sequence"].tolist()
    labels = dms_df["log_fluorescence"].values
    ln     = (labels - labels.mean()) / (labels.std() + 1e-9)

    idx_tr, idx_val = train_test_split(np.arange(len(seqs)), test_size=0.1,
                                        random_state=cfg["seed"])
    tr_dl  = _make_loader([seqs[i] for i in idx_tr],  ln[idx_tr],
                           tokenizer, cfg["max_length"], cfg["dms_batch_size"], True)
    val_dl = _make_loader([seqs[i] for i in idx_val], ln[idx_val],
                           tokenizer, cfg["max_length"], cfg["dms_batch_size"], False)

    opt   = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=cfg["dms_lr"], weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=cfg["dms_epochs"], eta_min=1e-6)
    loss_fn = nn.MSELoss()
    best_val, no_improve, best_state = float("inf"), 0, None

    for epoch in range(1, cfg["dms_epochs"] + 1):
        model.train()
        tr_loss = sum(
            _step(model, batch, opt, loss_fn, device, dms=True)
            for batch in tr_dl
        ) / len(tr_dl)

        model.eval()
        with torch.no_grad():
            vl_loss = sum(
                loss_fn(model(b["input_ids"].to(device),
                              b["attention_mask"].to(device))[0].squeeze(-1),
                        b["labels"].to(device)).item()
                for b in val_dl
            ) / len(val_dl)

        sched.step()
        log.info(f"  epoch {epoch:03d} | train {tr_loss:.4f} | val {vl_loss:.4f}")

        if vl_loss < best_val:
            best_val, no_improve = vl_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= cfg["dms_patience"]:
                log.info(f"  early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    log.info(f"  best val loss: {best_val:.4f}")
    return best_val


def _step(model, batch, opt, loss_fn, device, dms=False):
    ids, amsk, y = (batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["labels"].to(device))
    opt.zero_grad()
    pred, _ = model(ids, amsk)
    loss = loss_fn(pred.squeeze(-1), y) if dms else masked_mse(pred, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


def finetune_on_fpbase(model, seqs, tgt_scaled, idx_tr, idx_val,
                       tokenizer, device, cfg):
    """
    Stage 2: fine-tune on FPbase spectral data.
    LoRA adapters get a 10x lower LR than the head to preserve the DMS prior.
    """
    log.info(f"--- FPbase fine-tuning: {len(idx_tr)} train / {len(idx_val)} val ---")

    tr_dl  = _make_loader([seqs[i] for i in idx_tr],  tgt_scaled[idx_tr],
                           tokenizer, cfg["max_length"], cfg["fpbase_batch_size"], True)
    val_dl = _make_loader([seqs[i] for i in idx_val], tgt_scaled[idx_val],
                           tokenizer, cfg["max_length"], cfg["fpbase_batch_size"], False)

    lora_p = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    head_p = list(model.head.parameters())
    opt    = AdamW([{"params": lora_p, "lr": cfg["fpbase_lr"] * 0.1},
                    {"params": head_p, "lr": cfg["fpbase_lr"]}], weight_decay=0.01)
    sched  = CosineAnnealingLR(opt, T_max=cfg["fpbase_epochs"], eta_min=1e-6)
    best_val, no_improve, best_state = float("inf"), 0, None

    for epoch in range(1, cfg["fpbase_epochs"] + 1):
        model.train()
        tr_loss = sum(_step(model, b, opt, None, device) for b in tr_dl) / len(tr_dl)

        model.eval()
        with torch.no_grad():
            vl_loss = sum(
                masked_mse(model(b["input_ids"].to(device),
                                 b["attention_mask"].to(device))[0],
                           b["labels"].to(device)).item()
                for b in val_dl
            ) / len(val_dl)

        sched.step()
        if epoch % 5 == 0 or epoch == 1:
            log.info(f"  epoch {epoch:03d} | train {tr_loss:.4f} | val {vl_loss:.4f}")

        if vl_loss < best_val:
            best_val, no_improve = vl_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= cfg["fpbase_patience"]:
                log.info(f"  early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    log.info(f"  best val loss: {best_val:.4f}")
    return best_val


def extract_embeddings(model, seqs, tgt_scaled, idx_list, tokenizer, device, cfg):
    ds = SequenceDataset([seqs[i] for i in idx_list], tgt_scaled[idx_list],
                         tokenizer, cfg["max_length"])
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in dl:
            embs.append(
                model.get_embeddings(batch["input_ids"].to(device),
                                     batch["attention_mask"].to(device)).cpu().numpy()
            )
    return np.vstack(embs)
