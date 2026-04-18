import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _name_variants(name):
    base = name.lower().strip()
    return list(dict.fromkeys([
        base,
        base.replace(" ", "-"),
        base.replace(" ", "_"),
        base.replace(" ", ""),
        re.sub(r"[^a-z0-9\-]", "", base),
        re.sub(r"[^a-z0-9]",   "", base),
    ]))


def parse_ca_coords(pdb_path):
    coords, seen = [], set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[12:16].strip() != "CA":
                continue
            key = (line[21], line[22:26].strip())
            if key in seen:
                continue
            seen.add(key)
            coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"no CA atoms in {pdb_path}")
    return np.array(coords, dtype=np.float32)


def pocket_distances(ca, chromophore_res=66, radius=8.0, max_size=30):
    chrom = min(chromophore_res - 1, len(ca) - 1)
    dists = np.linalg.norm(ca - ca[chrom], axis=1)
    pocket = np.where(dists < radius)[0]
    if len(pocket) > max_size:
        pocket = pocket[np.argsort(dists[pocket])[:max_size]]

    feat = np.zeros((max_size, max_size), dtype=np.float32)
    k = len(pocket)
    if k > 1:
        pc   = ca[pocket]
        diff = pc[:, None] - pc[None]
        feat[:k, :k] = np.sqrt((diff**2).sum(-1)) / 50.0
    return feat.flatten()


def build_pocket_features(fpbase_df, structure_dir, cache_path,
                           chromophore_res=66, pocket_radius_a=8.0, max_pocket_size=30):
    cache = Path(cache_path)
    if cache.exists():
        feats = np.load(str(cache))
        if feats.shape[0] == len(fpbase_df):
            n = int((feats != 0).any(axis=1).sum())
            log.info(f"pocket features: cache hit {feats.shape}, {n}/{len(fpbase_df)} matched")
            return feats

    struct_dir = Path(structure_dir)
    feat_size  = max_pocket_size ** 2
    features   = np.zeros((len(fpbase_df), feat_size), dtype=np.float32)

    # build lookup: normalised stem -> Path
    pdb_lookup = {}
    for p in struct_dir.glob("*_minimized.pdb"):
        stem = p.stem.replace("_minimized", "")
        pdb_lookup[stem] = p
        pdb_lookup[re.sub(r"[^a-z0-9]", "", stem.lower())] = p

    n_pdbs = sum(1 for _ in struct_dir.glob("*_minimized.pdb"))
    log.info(f"structure dir: {struct_dir} ({n_pdbs} PDB files)")

    matched, unmatched = 0, []
    for i, row in fpbase_df.iterrows():
        name = str(row.get("name", ""))
        path = next((pdb_lookup[v] for v in _name_variants(name) if v in pdb_lookup), None)
        if path is None:
            unmatched.append(name)
            continue
        try:
            features[i] = pocket_distances(
                parse_ca_coords(path), chromophore_res, pocket_radius_a, max_pocket_size
            )
            matched += 1
        except Exception as e:
            log.warning(f"  [{i}] {name}: {e}")

    log.info(f"matched {matched}/{len(fpbase_df)} proteins to PDB structures")
    if unmatched:
        log.info(f"unmatched ({len(unmatched)}): {unmatched[:10]}")

    np.save(str(cache), features)
    return features
