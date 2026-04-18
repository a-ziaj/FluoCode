import logging
import requests
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

TARGETS = ["ex_max", "em_max", "ext_coeff", "qy"]


def fetch_fpbase(cache_path: str) -> pd.DataFrame:
    if Path(cache_path).exists():
        log.info(f"FPbase: loading cache from {cache_path}")
        return pd.read_csv(cache_path)

    log.info("FPbase: fetching from REST API...")
    url, proteins = "https://www.fpbase.org/api/proteins/?format=json&limit=2000", []
    while url:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            proteins.extend(data)
            break
        proteins.extend(data.get("results", []))
        url = data.get("next")

    rows = []
    for p in proteins:
        seq = p.get("seq", "")
        if not seq or len(seq) < 10:
            continue
        for s in (p.get("states", []) or [p]):
            rows.append({
                "name":      p.get("name", ""),
                "sequence":  seq,
                "ex_max":    s.get("ex_max"),
                "em_max":    s.get("em_max"),
                "ext_coeff": s.get("ext_coeff"),
                "qy":        s.get("qy"),
            })

    df = pd.DataFrame(rows)
    df["_n"] = df[TARGETS].notna().sum(axis=1)
    df = (df.sort_values("_n", ascending=False)
            .drop_duplicates("sequence")
            .drop(columns="_n")
            .reset_index(drop=True))

    if "qy" in df.columns:
        df.loc[~df["qy"].between(0, 1), "qy"] = np.nan
    if "ext_coeff" in df.columns:
        df.loc[df["ext_coeff"] <= 0, "ext_coeff"] = np.nan

    df.to_csv(cache_path, index=False)
    log.info(f"FPbase: {len(df)} proteins — {', '.join(f'{t}:{df[t].notna().sum()}' for t in TARGETS)}")
    return df


def load_fpbase(cache_path: str) -> pd.DataFrame:
    df = fetch_fpbase(cache_path)
    return df[df[TARGETS].notna().any(axis=1)].reset_index(drop=True)


def fetch_dms(cache_path: str, sample_n: int = 3000) -> pd.DataFrame:
    """
    Sarkisyan avGFP DMS dataset from HuggingFace.
    Stratified sample across brightness bins so the full dynamic range
    (non-fluorescent to bright) is covered.
    """
    if Path(cache_path).exists():
        log.info(f"DMS: loading cache from {cache_path}")
        return pd.read_csv(cache_path)

    log.info("DMS: fetching SaProtHub/Dataset-Fluorescence-TAPE...")
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("pip install datasets")

    raw = hf_load("SaProtHub/Dataset-Fluorescence-TAPE", split="train").to_pandas()
    raw = (raw.rename(columns={"protein": "sequence", "label": "log_fluorescence"})
              [["sequence", "log_fluorescence"]]
              .dropna()
              .query("sequence.str.len() > 10")
              .reset_index(drop=True))

    raw["_bin"] = pd.qcut(raw["log_fluorescence"], q=10, labels=False, duplicates="drop")
    n_bins = raw["_bin"].nunique()
    sampled = (raw.groupby("_bin", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), max(1, sample_n // n_bins)),
                                            random_state=42),
                         include_groups=False)
                  .reset_index(drop=True))
    sampled["orthologue"] = "avGFP"
    sampled.to_csv(cache_path, index=False)
    log.info(f"DMS: {len(sampled)} sequences sampled from {len(raw)} ({n_bins} brightness bins)")
    return sampled


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences  = sequences
        self.labels     = torch.tensor(np.array(labels), dtype=torch.float32)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.sequences[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         self.labels[idx],
        }
