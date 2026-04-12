import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalTransformer(nn.Module):
    """
    Dual-stream Multi-Modal deep learning architecture integrating spatial graph topologies
    (via Graph Transformer) and sequential molecular motifs (via MLP on Morgan fingerprints).

    Args:
        node_feat_dim (int): Dimensionality of initial node (atom) features. Default: 43.
        hidden_dim (int): Dimensionality of latent embeddings. Default: 512.
        fp_dim (int): Dimensionality of the input molecular fingerprint. Default: 1024.
        output_dim (int): Dimensionality of the target transcriptomic perturbation vector. Default: 12328.
    """

    def __init__(self, node_feat_dim=43, hidden_dim=512, fp_dim=1024, output_dim=12328):
        super(MultiModalTransformer, self).__init__()

        # ==========================================
        # Modality 1: Spatial Topology (Graph Stream)
        # ==========================================
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)

        # Multi-head self-attention mechanism for graph representation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.graph_post_pool = nn.Linear(hidden_dim, hidden_dim)

        # ==========================================
        # Modality 2: Sequential Motif (Fingerprint Stream)
        # ==========================================
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, hidden_dim),
            nn.ReLU()
        )

        # ==========================================
        # Cross-Modal Fusion and Decoding Head
        # ==========================================
        # Concatenated latent space: hidden_dim (Graph) + hidden_dim (FP)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, output_dim)
        )

    def forward(self, x, adj, mask, fingerprint):
        """
        Forward propagation pass.

        Args:
            x (Tensor): Node feature matrix (Batch, Max_Atoms, Node_Feat_Dim).
            adj (Tensor): Adjacency matrix (Batch, Max_Atoms, Max_Atoms) - Reserved for explicit message passing if needed.
            mask (Tensor): Validity mask for padding (Batch, Max_Atoms, 1).
            fingerprint (Tensor): Molecular fingerprint vector (Batch, FP_Dim).

        Returns:
            prediction (Tensor): Predicted transcriptomic signature vector (Batch, Output_Dim).
        """
        # --- 1. Graph Stream Processing ---
        h = self.node_embedding(x)

        # Generate padding mask for Transformer (True indicates padding tokens to be ignored)
        key_padding_mask = (mask.squeeze(-1) == 0)
        h_trans = self.transformer_encoder(h, src_key_padding_mask=key_padding_mask)

        # Apply mask and perform global mean pooling
        h_trans = h_trans * mask
        sum_h = torch.sum(h_trans, dim=1)
        num_atoms = torch.sum(mask, dim=1)
        graph_feat = sum_h / (num_atoms + 1e-6)
        graph_feat = self.graph_post_pool(graph_feat)

        # --- 2. Fingerprint Stream Processing ---
        fp_feat = self.fp_encoder(fingerprint)

        # --- 3. Representation Fusion and Decoding ---
        combined = torch.cat([graph_feat, fp_feat], dim=1)
        prediction = self.fusion_layer(combined)

        return prediction