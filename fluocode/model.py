import torch
import torch.nn as nn
from transformers import EsmModel
from peft import LoraConfig, get_peft_model, TaskType


class ESM2Regressor(nn.Module):
    """
    ESM2 with LoRA adapters and a small MLP head.
    Used with output_dim=1 for DMS brightness pretraining,
    then rebuilt with output_dim=4 for FPbase spectral prediction.
    """

    def __init__(self, model_name, lora_cfg, output_dim=1, dropout=0.15):
        super().__init__()
        self.encoder = get_peft_model(EsmModel.from_pretrained(model_name), lora_cfg)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def _pool(self, input_ids, attention_mask):
        h    = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(self, input_ids, attention_mask):
        pooled = self._pool(input_ids, attention_mask)
        return self.head(pooled), pooled

    def get_embeddings(self, input_ids, attention_mask):
        with torch.no_grad():
            return self._pool(input_ids, attention_mask)

    def n_trainable(self):
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n = sum(p.numel() for p in self.parameters())
        return t, n


def make_lora_config(r, alpha, dropout, target_modules=None):
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or ["query", "key", "value"],
        bias="none",
    )


def transfer_base_weights(src, dst):
    """Copy ESM2 base weights from src to dst, skipping LoRA matrices."""
    base = {k: v for k, v in src.encoder.state_dict().items()
            if "lora_A" not in k and "lora_B" not in k}
    dst.encoder.load_state_dict(base, strict=False)
    return len(base)
