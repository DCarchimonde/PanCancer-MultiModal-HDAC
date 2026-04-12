import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MultiModalDataset
from multimodal_model import MultiModalTransformer
import warnings
import os

# 忽略警告
warnings.simplefilter(action='ignore', category=FutureWarning)


# ==========================================
# 🛠️ 核心工具：多模态填充函数 (Collate)
# ==========================================
def pad_collate_multimodal(batch):
    # 1. 过滤 None
    batch = [item for item in batch if item is not None]
    if len(batch) == 0: return None

    # 2. 找出最大原子数
    max_atoms = max([item[0].shape[0] for item in batch])

    batch_x = []
    batch_adj = []
    batch_mask = []
    batch_fp = []
    batch_y = []

    for item in batch:
        x, adj, fp, y = item
        num_atoms = x.shape[0]

        # 填充 X
        x_padded = torch.zeros((max_atoms, x.shape[1]))
        x_padded[:num_atoms, :] = x
        batch_x.append(x_padded)

        # 填充 Adj
        adj_padded = torch.zeros((max_atoms, max_atoms))
        adj_padded[:num_atoms, :num_atoms] = adj
        batch_adj.append(adj_padded)

        # 填充 Mask
        mask = torch.zeros((max_atoms, 1))
        mask[:num_atoms, :] = 1
        batch_mask.append(mask)

        # 指纹
        batch_fp.append(fp)

        # Y
        batch_y.append(y)

    return (torch.stack(batch_x),
            torch.stack(batch_adj),
            torch.stack(batch_mask),
            torch.stack(batch_fp),
            torch.stack(batch_y))


# ==========================================
# 🚀 主程序
# ==========================================
if __name__ == '__main__':
    # 1. 配置
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0005
    EPOCHS = 60
    DEBUG_MODE = False
    NUM_WORKERS = 12  # 稍微保守一点，防止显存抢不过CPU

    # 🔥 关键配置：续训文件路径
    RESUME_CHECKPOINT = 'multimodal_model_epoch_36.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 多模态引擎启动: {device} | Batch: {BATCH_SIZE}")

    # 2. 数据集加载
    csv_path = './data/clean_dataset.csv'
    gctx_path = './data/level5_beta_trt_cp_n720216x12328.gctx'
    metrics_path = './data/GSE92742_Broad_LINCS_sig_metrics.txt'

    print("\n>>> 正在加载多模态数据集...")
    if not os.path.exists(metrics_path):
        print("⚠️ 没找到 metrics 文件，暂时跳过 TAS 过滤...")
        dataset = MultiModalDataset(csv_path, gctx_path, metrics_file=None, debug_mode=DEBUG_MODE)
    else:
        dataset = MultiModalDataset(csv_path, gctx_path, metrics_file=metrics_path, debug_mode=DEBUG_MODE)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 续训时Shuffle依然要是True
        collate_fn=pad_collate_multimodal,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True
    )

    print(">>> 正在初始化 Transformer + MLP 模型...")
    # 确保这里的参数和你训练 epoch 13 时完全一致！
    model = MultiModalTransformer(node_feat_dim=43, hidden_dim=128, output_dim=12328)
    model = model.to(device)

    # ==========================================
    # 🔄 续训逻辑：加载权重 & 设置起始轮数
    # ==========================================
    start_epoch = 0  # 默认从 0 开始

    if os.path.exists(RESUME_CHECKPOINT):
        print(f"\nFound Checkpoint: {RESUME_CHECKPOINT}")
        print("🔄 正在加载存档，恢复训练进度...")

        # 加载权重
        checkpoint = torch.load(RESUME_CHECKPOINT)
        model.load_state_dict(checkpoint)

        # 自动解析文件名里的轮数，比如 '...epoch_13.pth' -> 13
        # 如果文件名格式不对，请手动把 start_epoch 改成 13
        try:
            # 假设文件名格式是 xxx_epoch_13.pth
            loaded_epoch = int(RESUME_CHECKPOINT.split('_epoch_')[1].split('.pth')[0])
            start_epoch = loaded_epoch  # 下一轮循环 range 会从 start_epoch 开始 (即 13->29，实际跑的是第14轮)
            # 注意：如果 epoch_13 代表跑完了 13 轮（0~12），那这里应该是 13
            # 如果 epoch_13 代表跑完了第 13 轮（index 12），通常我们理解为这就是当前的进度
            # 为了保险，我们假设你跑完了 13 轮，现在要跑第 14 轮 (Index 13)
        except:
            print("⚠️ 无法从文件名解析轮数，默认设置为 13 (即从第14轮开始跑)")
            start_epoch = 13

        print(f"✅ 存档加载成功！将从 Epoch {start_epoch + 1} 开始继续训练...")
    else:
        print(f"❌ 未找到存档 {RESUME_CHECKPOINT}，将从头开始训练 (Epoch 1)...")

    # ==========================================

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    print(f"\n>>> 开始训练！目标跑完 {EPOCHS} 轮")

    # 🔥 修改 range，从 start_epoch 开始
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0

        # 进度显示 (为了让你安心，加个 total 打印)
        print(f"--> 进入 Epoch {epoch + 1}...")

        for i, data in enumerate(train_loader):
            if data is None: continue

            x, adj, mask, fp, y = data

            x = x.to(device, non_blocking=True)
            adj = adj.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            fp = fp.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            prediction = model(x, adj, mask, fp)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:  # 改成 50 打印一次，不然刷屏太快
                print(f"  [Epoch {epoch + 1}, Batch {i + 1}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"=== Epoch {epoch + 1} 结束 | 平均 Loss: {avg_loss:.4f} ===\n")

        # 保存时建议统一命名格式，方便下次再续
        # 这里我保留了你原来的命名风格，但建议用 multimodal_model
        save_name = f'multimodal_model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), save_name)
        print(f"💾 已保存模型: {save_name}")

    print(">>> 训练全部完成！")