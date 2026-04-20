import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool


class PocketGNN(nn.Module):
    """
    Graph attention network over the chromophore pocket.

    Each protein's pocket is a small graph:
      nodes  — residues within radius A of the chromophore
      edges  — pairs of residues within edge_cutoff A of each other
      node features — ESM2 per-residue embeddings at pocket positions (480-dim)
      edge features — normalised Cα distance between the two residues

    Three GATv2Conv layers with residual connections, followed by global
    mean pooling to produce a fixed-size pocket embedding regardless of
    how many residues are in the pocket.
    """

    def __init__(self, node_dim: int = 480, hidden: int = 256,
                 out_dim: int = 128, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATv2Conv(node_dim,  hidden // heads, heads=heads,
                                edge_dim=1, dropout=dropout)
        self.conv2 = GATv2Conv(hidden, hidden // heads, heads=heads,
                                edge_dim=1, dropout=dropout)
        self.conv3 = GATv2Conv(hidden, out_dim,          heads=1,
                                edge_dim=1, dropout=dropout)

        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.norm3 = nn.LayerNorm(out_dim)
        self.act   = nn.GELU()
        self.drop  = nn.Dropout(dropout)

        # project node features to hidden dim for residual connections
        self.res_proj = nn.Linear(node_dim, hidden, bias=False)

    def forward(self, x, edge_index, edge_attr, batch):
        # layer 1
        h  = self.act(self.norm1(self.conv1(x, edge_index, edge_attr)))
        h  = self.drop(h) + self.res_proj(x)           # residual

        # layer 2
        h2 = self.act(self.norm2(self.conv2(h, edge_index, edge_attr)))
        h2 = self.drop(h2) + h                          # residual

        # layer 3
        h3 = self.norm3(self.conv3(h2, edge_index, edge_attr))

        return global_mean_pool(h3, batch)               # [n_graphs, out_dim]


def build_pocket_graph(ca_coords, residue_embs, chrom_res=66,
                       pocket_radius=8.0, edge_cutoff=6.0):
    """
    Build a PyG Data object for one protein's chromophore pocket.

    Args:
        ca_coords:    float32 [L, 3] — Cα coordinates for all residues
        residue_embs: float32 [L, 480] — ESM2 per-residue embeddings
        chrom_res:    1-indexed chromophore anchor residue
        pocket_radius: residues within this distance of chrom_res form the pocket
        edge_cutoff:  edges connect residues within this distance of each other

    Returns:
        torch_geometric.data.Data with x, edge_index, edge_attr set,
        or None if the pocket has fewer than 2 residues.
    """
    L         = len(ca_coords)
    chrom_idx = min(chrom_res - 1, L - 1)
    dists     = torch.norm(ca_coords - ca_coords[chrom_idx], dim=1)
    pocket    = (dists < pocket_radius).nonzero(as_tuple=True)[0]

    if len(pocket) < 2:
        return None

    pocket_ca   = ca_coords[pocket]       # [k, 3]
    pocket_embs = residue_embs[pocket]    # [k, 480]

    # build edges: connect all pairs within edge_cutoff
    k    = len(pocket)
    diff = pocket_ca.unsqueeze(0) - pocket_ca.unsqueeze(1)   # [k, k, 3]
    dist_mat = torch.norm(diff, dim=2)                        # [k, k]

    src, dst = (dist_mat < edge_cutoff).nonzero(as_tuple=True)
    # remove self-loops
    mask = src != dst
    src, dst = src[mask], dst[mask]

    if len(src) == 0:
        # fallback: fully connect the pocket
        idx  = torch.arange(k)
        src  = idx.repeat_interleave(k)
        dst  = idx.repeat(k)
        mask = src != dst
        src, dst = src[mask], dst[mask]

    edge_attr = dist_mat[src, dst].unsqueeze(1) / 50.0   # normalise by 50A

    return Data(
        x          = pocket_embs,
        edge_index = torch.stack([src, dst]),
        edge_attr  = edge_attr,
    )


def extract_residue_embeddings(model, seqs, tokenizer, device, cfg):
    """
    Extract per-residue ESM2 embeddings (before mean pooling).
    Returns a list of float32 tensors, one per sequence, shape [L, 480].
    """
    from torch.utils.data import DataLoader
    from fluocode.data import SequenceDataset
    import numpy as np

    # dummy labels — we only need the embeddings
    dummy = np.zeros((len(seqs), 1), dtype=np.float32)
    ds    = SequenceDataset(seqs, dummy, tokenizer, cfg["max_length"])
    dl    = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    model.eval()
    all_embs = []
    with torch.no_grad():
        for batch in dl:
            ids  = batch["input_ids"].to(device)
            amsk = batch["attention_mask"].to(device)
            out  = model.encoder(input_ids=ids, attention_mask=amsk)
            h    = out.last_hidden_state   # [B, L, 480]
            # strip padding — return variable-length tensors
            for b_idx in range(h.shape[0]):
                seq_len = amsk[b_idx].sum().item()
                all_embs.append(h[b_idx, 1:seq_len-1].cpu())  # strip [CLS]/[EOS]
    return all_embs
