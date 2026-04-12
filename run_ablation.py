import os

# Resolve potential library conflicts on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
from torch.utils.data import DataLoader
from dataset import MultiModalDataset
from scipy.stats import pearsonr
from tqdm import tqdm

warnings.filterwarnings("ignore")


def pad_collate_multimodal(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    max_atoms = max([item[0].shape[0] for item in batch])
    bx, badj, bmask, bfp, by = [], [], [], [], []
    for item in batch:
        x, adj, fp, y = item
        num = x.shape[0]
        xp = torch.zeros((max_atoms, x.shape[1]))
        xp[:num, :] = x
        bx.append(xp)
        ap = torch.zeros((max_atoms, max_atoms))
        ap[:num, :num] = adj
        badj.append(ap)
        bp = torch.zeros((max_atoms, 1))
        bp[:num, :] = 1
        bmask.append(bp)
        bfp.append(fp)
        by.append(y)
    return (torch.stack(bx), torch.stack(badj), torch.stack(bmask), torch.stack(bfp), torch.stack(by))


def calculate_profile_wise_pearson(y_true_matrix, y_pred_matrix):
    p_list = []
    for i in range(len(y_true_matrix)):
        p_r, _ = pearsonr(y_true_matrix[i], y_pred_matrix[i])
        if not np.isnan(p_r): p_list.append(p_r)
    return np.mean(p_list)


# Ablation Model 1: GNN Only (without functional fingerprint)
class GNN_Only(nn.Module):
    def __init__(self, node_feat_dim=43, hidden_dim=512, output_dim=12328):
        super().__init__()
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fusion = nn.Sequential(nn.Linear(hidden_dim, 2048), nn.ReLU(), nn.Linear(2048, output_dim))

    def forward(self, x, adj, mask, fp):
        h = self.node_embedding(x)
        key_padding_mask = (mask.squeeze(-1) == 0)
        h_trans = self.transformer_encoder(h, src_key_padding_mask=key_padding_mask)
        h_trans = h_trans * mask
        graph_feat = torch.sum(h_trans, dim=1) / (torch.sum(mask, dim=1) + 1e-6)
        return self.fusion(graph_feat)


# Ablation Model 2: GNN + Fingerprint via simple concatenation (without cross-attention)
class GNN_Concat_FP(nn.Module):
    def __init__(self, node_feat_dim=43, hidden_dim=512, fp_dim=1024, output_dim=12328):
        super().__init__()
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fp_encoder = nn.Sequential(nn.Linear(fp_dim, 1024), nn.ReLU(), nn.Linear(1024, hidden_dim), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(hidden_dim * 2, 2048), nn.ReLU(), nn.Linear(2048, output_dim))

    def forward(self, x, adj, mask, fp):
        h = self.node_embedding(x)
        key_padding_mask = (mask.squeeze(-1) == 0)
        h_trans = self.transformer_encoder(h, src_key_padding_mask=key_padding_mask)
        h_trans = h_trans * mask
        graph_feat = torch.sum(h_trans, dim=1) / (torch.sum(mask, dim=1) + 1e-6)
        fp_feat = self.fp_encoder(fp)
        return self.fusion(torch.cat([graph_feat, fp_feat], dim=1))


if __name__ == '__main__':
    print("Starting ablation study...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MultiModalDataset('./data/train_dataset_drugcell.csv', './data/level5_beta_trt_cp_n720216x12328.gctx',
                                 metrics_file=None)
    test_ds = MultiModalDataset('./data/test_dataset_drugcell.csv', './data/level5_beta_trt_cp_n720216x12328.gctx',
                                metrics_file=None)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=pad_collate_multimodal, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=pad_collate_multimodal, num_workers=0)

    models_to_test = {
        "1_GNN_Only": GNN_Only().to(device),
        "2_GNN_Concat": GNN_Concat_FP().to(device)
    }

    results = {}

    for name, model in models_to_test.items():
        print(f"\nTraining ablation model: {name}")
        optimizer = optim.AdamW(model.parameters(), lr=0.0005)
        criterion = nn.MSELoss()

        # Train for 5 epochs to observe trends
        for epoch in range(5):
            model.train()
            for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/5"):
                if data is None: continue
                x, adj, mask, fp, y = [d.to(device) for d in data]
                optimizer.zero_grad()
                loss = criterion(model(x, adj, mask, fp), y)
                loss.backward()
                optimizer.step()

        print(f"Testing {name}...")
        model.eval()
        all_y_true, all_y_pred = [], []
        with torch.no_grad():
            for data in tqdm(test_loader, desc="Testing"):
                if data is None: continue
                x, adj, mask, fp, y = [d.to(device) for d in data]
                pred = model(x, adj, mask, fp)
                all_y_true.append(y.cpu().numpy())
                all_y_pred.append(pred.cpu().numpy())

        y_true_arr = np.concatenate(all_y_true)
        y_pred_arr = np.concatenate(all_y_pred)

        # Save predictions for evaluation
        np.save(f'y_true_{name}.npy', y_true_arr)
        np.save(f'y_pred_{name}.npy', y_pred_arr)
        print(f"Results saved for {name}.")

        print(f"Calculating Pearson correlation for {name}...")
        r = calculate_profile_wise_pearson(y_true_arr, y_pred_arr)
        results[name] = r
        print(f"{name} Pearson R: {r:.4f}")

    print("\n" + "=" * 50)
    print("Ablation Study Results")
    print("=" * 50)
    for name, r in results.items():
        print(f"{name}: Pearson R = {r:.4f}")
    print("Full Model (Cross-Attention): Pearson R = 0.2841")
    print("=" * 50)