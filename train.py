import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MultiModalDataset
from multimodal_model import MultiModalTransformer

# Suppress warnings for clean output console
warnings.simplefilter(action='ignore', category=FutureWarning)

def pad_collate_multimodal(batch):
    """
    Custom collate function to dynamically pad graph nodes and adjacency matrices
    to the maximum graph size within the current batch.
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    max_atoms = max([item[0].shape[0] for item in batch])
    batch_x, batch_adj, batch_mask, batch_fp, batch_y = [], [], [], [], []

    for item in batch:
        x, adj, fp, y = item
        num_atoms = x.shape[0]

        # Pad node features (X)
        x_padded = torch.zeros((max_atoms, x.shape[1]))
        x_padded[:num_atoms, :] = x
        batch_x.append(x_padded)

        # Pad adjacency matrix (Adj)
        adj_padded = torch.zeros((max_atoms, max_atoms))
        adj_padded[:num_atoms, :num_atoms] = adj
        batch_adj.append(adj_padded)

        # Create attention mask
        mask = torch.zeros((max_atoms, 1))
        mask[:num_atoms, :] = 1
        batch_mask.append(mask)

        batch_fp.append(fp)
        batch_y.append(y)

    return (torch.stack(batch_x),
            torch.stack(batch_adj),
            torch.stack(batch_mask),
            torch.stack(batch_fp),
            torch.stack(batch_y))

if __name__ == '__main__':
    # --- Hyperparameters & Configuration ---
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0005
    EPOCHS = 60
    DEBUG_MODE = False
    NUM_WORKERS = 12
    RESUME_CHECKPOINT = 'multimodal_model_epoch_36.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Multi-Modal Pipeline | Device: {device} | Batch Size: {BATCH_SIZE}")

    # --- Data Loading ---
    csv_path = './data/clean_dataset.csv'
    gctx_path = './data/level5_beta_trt_cp_n720216x12328.gctx'
    metrics_path = './data/GSE92742_Broad_LINCS_sig_metrics.txt'

    print("Loading multi-modal dataset...")
    metrics_file = metrics_path if os.path.exists(metrics_path) else None
    dataset = MultiModalDataset(csv_path, gctx_path, metrics_file=metrics_file, debug_mode=DEBUG_MODE)

    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=pad_collate_multimodal,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True
    )

    # --- Model Initialization ---
    print("Initializing Dual-Stream GNN-Transformer Architecture...")
    model = MultiModalTransformer(node_feat_dim=43, hidden_dim=128, output_dim=12328)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # --- Checkpoint Resumption ---
    start_epoch = 0
    if os.path.exists(RESUME_CHECKPOINT):
        print(f"Loading checkpoint from: {RESUME_CHECKPOINT}")
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location=device)
        model.load_state_dict(checkpoint)
        try:
            start_epoch = int(RESUME_CHECKPOINT.split('_epoch_')[1].split('.pth')[0])
        except ValueError:
            print("Warning: Could not parse epoch from filename. Starting from default index.")
        print(f"Successfully loaded. Resuming training from Epoch {start_epoch + 1}...")
    else:
        print("No checkpoint found. Initiating training from scratch (Epoch 1)...")

    # --- Training Loop ---
    print(f"Training initiated. Target Epochs: {EPOCHS}")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0

        print(f"--- Epoch {epoch + 1}/{EPOCHS} ---")
        for i, data in enumerate(train_loader):
            if data is None: continue

            x, adj, mask, fp, y = data
            x, adj, mask = x.to(device, non_blocking=True), adj.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            fp, y = fp.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad()
            prediction = model(x, adj, mask, fp)
            loss = criterion(prediction, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"   [Batch {i + 1}] Iteration Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"End of Epoch {epoch + 1} | Average Loss: {avg_loss:.4f}\n")

        save_name = f'multimodal_model_epoch_{epoch + 1}.pth'
        torch.save(model.state_dict(), save_name)
        print(f"Checkpoint saved: {save_name}")

    print("Training successfully completed.")