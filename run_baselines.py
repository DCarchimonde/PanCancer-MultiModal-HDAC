import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import warnings
from torch.utils.data import DataLoader
from dataset import MultiModalDataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

warnings.filterwarnings("ignore")


# 基础数据加载 (复用)
def pad_collate_multimodal(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None
    max_atoms = max([item[0].shape[0] for item in batch])
    bx, badj, bmask, bfp, by = [], [], [], [], []
    for item in batch:
        x, adj, fp, y = item;
        num = x.shape[0]
        xp = torch.zeros((max_atoms, x.shape[1]));
        xp[:num, :] = x;
        bx.append(xp)
        ap = torch.zeros((max_atoms, max_atoms));
        ap[:num, :num] = adj;
        badj.append(ap)
        bp = torch.zeros((max_atoms, 1));
        bp[:num, :] = 1;
        bmask.append(bp)
        bfp.append(fp);
        by.append(y)
    return (torch.stack(bx), torch.stack(badj), torch.stack(bmask), torch.stack(bfp), torch.stack(by))


def calculate_profile_wise_pearson(y_true_matrix, y_pred_matrix):
    p_list = []
    for i in range(len(y_true_matrix)):
        r, _ = pearsonr(y_true_matrix[i], y_pred_matrix[i])
        if not np.isnan(r): p_list.append(r)
    return np.mean(p_list)


# 🗡️ SOTA 1: DeepCE Proxy (纯图卷积 GNN 映射)
class DeepCE_Proxy(nn.Module):
    def __init__(self, node_feat_dim=43, hidden_dim=256, output_dim=12328):
        super().__init__()
        self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
        self.gcn1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, 1024), nn.ReLU(), nn.Linear(1024, output_dim))

    def forward(self, x, adj, mask, fp):
        h = torch.relu(self.node_embedding(x))
        h = torch.relu(self.gcn1(torch.bmm(adj, h)))
        h = torch.relu(self.gcn2(torch.bmm(adj, h)))
        h = h * mask
        graph_feat = torch.sum(h, dim=1) / (torch.sum(mask, dim=1) + 1e-6)
        return self.decoder(graph_feat)


# 🗡️ SOTA 2: ChemCPA Proxy (纯指纹深度编码器映射)
class ChemCPA_Proxy(nn.Module):
    def __init__(self, fp_dim=1024, latent_dim=512, output_dim=12328):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, latent_dim), nn.BatchNorm1d(latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048), nn.ReLU(), nn.Dropout(0.2), nn.Linear(2048, output_dim)
        )

    def forward(self, x, adj, mask, fp):
        latent = self.encoder(fp)
        return self.decoder(latent)


if __name__ == '__main__':
    print("🚀 启动 [双重 SOTA 联合复现机 - 绝对安全版]...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MultiModalDataset('./data/train_dataset_drugcell.csv', './data/level5_beta_trt_cp_n720216x12328.gctx',
                                 metrics_file=None)
    test_ds = MultiModalDataset('./data/test_dataset_drugcell.csv', './data/level5_beta_trt_cp_n720216x12328.gctx',
                                metrics_file=None)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=pad_collate_multimodal, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, collate_fn=pad_collate_multimodal, num_workers=0)

    models_to_test = {
        "DeepCE_Proxy": DeepCE_Proxy().to(device),
        "ChemCPA_Proxy": ChemCPA_Proxy().to(device)
    }

    results = {}

    for name, model in models_to_test.items():
        print(f"\n🔬 正在特训 SOTA 模型: {name}")
        optimizer = optim.AdamW(model.parameters(), lr=0.0005)
        criterion = nn.MSELoss()

        # 只跑 5 轮即可看清天花板
        for epoch in range(5):
            model.train()
            for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/5"):
                if data is None: continue
                x, adj, mask, fp, y = [d.to(device) for d in data]
                optimizer.zero_grad()
                loss = criterion(model(x, adj, mask, fp), y)
                loss.backward()
                optimizer.step()

            # 🔥 安全感拉满：每一轮都保存 SOTA 的权重！
            torch.save(model.state_dict(), f"{name}_epoch_{epoch + 1}.pth")

        print(f"📊 正在测试 {name}...")
        model.eval()
        all_y_true, all_y_pred = [], []
        with torch.no_grad():
            for data in tqdm(test_loader, desc="全量推理中"):
                if data is None: continue
                x, adj, mask, fp, y = [d.to(device) for d in data]
                pred = model(x, adj, mask, fp)
                all_y_true.append(y.cpu().numpy())
                all_y_pred.append(pred.cpu().numpy())

        y_true = np.concatenate(all_y_true)
        y_pred = np.concatenate(all_y_pred)

        # 🔥 再次吃下后悔药：保存预测结果数组
        np.save(f'y_true_{name}.npy', y_true)
        np.save(f'y_pred_{name}.npy', y_pred)

        # 计算 Pearson R
        r = calculate_profile_wise_pearson(y_true, y_pred)
        # 💥 新增计算 MSE
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())

        results[name] = {"R": r, "MSE": mse}
        print(f"✅ {name} 最终成绩 -> Pearson R: {r:.4f} | MSE: {mse:.4f}")

    print("\n" + "=" * 50)
    print("👑 【新增 SOTA 最终成绩单】")
    print("=" * 50)
    for name, metrics in results.items():
        print(f"📉 {name}: MSE = {metrics['MSE']:.4f}, R = {metrics['R']:.4f}")
    print(f"👉 将这些数字填入你的 Table 1 中吧！")
    print("=" * 50)