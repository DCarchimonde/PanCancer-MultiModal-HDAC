from __future__ import annotations

import torch
from torch import nn


class AtomTokenFingerprintModel(nn.Module):
    """Atom-token Transformer stream plus Morgan-fingerprint MLP stream."""

    def __init__(
        self,
        node_feature_dim: int = 43,
        fingerprint_dim: int = 1024,
        hidden_dim: int = 512,
        output_dim: int = 12328,
        transformer_layers: int = 3,
        attention_heads: int = 8,
    ) -> None:
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")

        self.node_embedding = nn.Linear(node_feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        self.atom_post_pool = nn.Linear(hidden_dim, hidden_dim)

        self.fingerprint_encoder = nn.Sequential(
            nn.Linear(fingerprint_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, output_dim),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.node_embedding(atom_features)
        encoded = self.transformer_encoder(
            encoded,
            src_key_padding_mask=~atom_mask,
        )
        mask = atom_mask.unsqueeze(-1).to(encoded.dtype)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        atom_representation = self.atom_post_pool(pooled)
        fingerprint_representation = self.fingerprint_encoder(fingerprint)
        return self.fusion_head(
            torch.cat([atom_representation, fingerprint_representation], dim=1)
        )


class FingerprintMLP(nn.Module):
    """Capacity-matched Morgan-fingerprint baseline."""

    def __init__(
        self,
        fingerprint_dim: int = 1024,
        output_dim: int = 12328,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(fingerprint_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, output_dim),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        atom_mask: torch.Tensor,
        fingerprint: torch.Tensor,
    ) -> torch.Tensor:
        del atom_features, atom_mask
        return self.network(fingerprint)


def build_model(model_name: str, hidden_dim: int, output_dim: int) -> nn.Module:
    if model_name == "dual_stream":
        return AtomTokenFingerprintModel(hidden_dim=hidden_dim, output_dim=output_dim)
    if model_name == "fingerprint_mlp":
        return FingerprintMLP(output_dim=output_dim)
    raise ValueError(f"Unknown model: {model_name}")
