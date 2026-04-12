import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalTransformer(nn.Module):
    # 🔥 升级点 1：把 hidden_dim 默认值提到了 512，脑容量扩大 4 倍！
    def __init__(self, node_feat_dim=43, hidden_dim=512, fp_dim=1024, output_dim=12328):
        super(MultiModalTransformer, self).__init__()

        # ==========================
        # 🏗️ 左塔: Graph Transformer
        # ==========================
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)  # 增加到 8 头注意力
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)  # 增加到 3 层，更深
        self.graph_post_pool = nn.Linear(hidden_dim, hidden_dim)

        # ==========================
        # 🏗️ 右塔: Fingerprint MLP
        # ==========================
        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, hidden_dim),
            nn.ReLU()
        )

        # ==========================
        # 🔗 融合层 (Fusion)
        # ==========================
        # 输入: Graph(512) + FP(512) = 1024
        # 🔥 升级点 2：中间过渡层放大到 2048，防止 1024 直接暴增到 12328 导致信息丢失
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(2048, output_dim)
        )

    def forward(self, x, adj, mask, fingerprint):
        # --- 1. 左塔 ---
        h = self.node_embedding(x)
        key_padding_mask = (mask.squeeze(-1) == 0)
        h_trans = self.transformer_encoder(h, src_key_padding_mask=key_padding_mask)

        h_trans = h_trans * mask
        sum_h = torch.sum(h_trans, dim=1)
        num_atoms = torch.sum(mask, dim=1)
        graph_feat = sum_h / (num_atoms + 1e-6)
        graph_feat = self.graph_post_pool(graph_feat)

        # --- 2. 右塔 ---
        fp_feat = self.fp_encoder(fingerprint)

        # --- 3. 融合 ---
        combined = torch.cat([graph_feat, fp_feat], dim=1)
        prediction = self.fusion_layer(combined)

        return prediction