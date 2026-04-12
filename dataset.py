import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from cmapPy.pandasGEXpress.parse import parse
from drug_to_graph import smiles_to_graph  # 引用你之前的工具
from rdkit.Chem import rdFingerprintGenerator


class MultiModalDataset(Dataset):
    def __init__(self, csv_file, gctx_file, metrics_file=None, debug_mode=False):
        """
        参数:
        csv_file: 你的 clean_dataset.csv
        gctx_file: 那个 30G 的大文件
        metrics_file: 刚下载的 sig_metrics.txt (用于质量过滤)
        """
        self.gctx_file = gctx_file

        print(f"正在读取主索引: {csv_file}")
        self.meta_data = pd.read_csv(csv_file)

        # ==========================================
        # 🟢 Task 2: 质量控制与金标准过滤
        # ==========================================
        if metrics_file:
            print(f"正在读取质量文件: {metrics_file} ...")
            # sig_metrics.txt 比较大，我们要根据 sig_id 来合并
            metrics_df = pd.read_csv(metrics_file, sep='\t', usecols=['sig_id', 'tas', 'distil_cc_q75'])

            # 合并：给 clean_dataset 加上 tas 分数
            original_len = len(self.meta_data)
            self.meta_data = pd.merge(self.meta_data, metrics_df, on='sig_id', how='inner')
            print(f"合并元数据后: {len(self.meta_data)} 条 (原始: {original_len})")

            # 过滤：只保留 TAS >= 0.2 的数据 (这是 CMap 推荐的阈值)
            # TAS < 0.2 说明药物没起作用，或者是噪音
            self.meta_data = self.meta_data[self.meta_data['tas'] >= 0.2]
            print(f"🧹 过滤低质量数据 (TAS < 0.2) 后剩余: {len(self.meta_data)} 条")
        else:
            print("⚠️ 警告: 未提供 metrics_file，将跳过质量过滤！(仅用于测试)")

        # ==========================================
        # 🧹 新增: 剔除 'restricted' 的保密数据
        # ==========================================
        if 'smiles' in self.meta_data.columns:
            # 记录一下之前的数量
            before_clean = len(self.meta_data)

            # 只保留那些 SMILES 不是 'restricted' 的行
            self.meta_data = self.meta_data[self.meta_data['smiles'] != 'restricted']

            print(f"🧹 剔除 SMILES='restricted' 数据: {before_clean} -> {len(self.meta_data)}")

        # --- Debug 模式 ---
        if debug_mode:
            print("⚠️ DEBUG模式: 只取前 200 条")
            self.meta_data = self.meta_data.head(200)

        # 缓存列表，加速读取
        self.all_sig_ids = self.meta_data['sig_id'].tolist()
        self.all_smiles = self.meta_data['smiles'].tolist()  # 确保你的csv里列名叫 smiles

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        # 1. 获取基本信息
        smiles = self.all_smiles[idx]
        sig_id = self.all_sig_ids[idx]

        # 2. 模态一: 图特征 (Graph)
        features, adj = smiles_to_graph(smiles)
        if features is None or adj is None:
            return None

        # 3. 模态二: 化学指纹 (Fingerprint) - NEW! ✨
        # 这是一个 1024 维的 0/1 向量，捕捉官能团信息
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 半径2，1024位 (这是工业界标准配置)
        # 创建生成器 (半径2, 1024位)
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        # 直接生成 NumPy 数组 (浮点型)
        fp_array = morgan_gen.GetFingerprintAsNumPy(mol)

        # 4. 获取标签 (Gene Expression)
        try:
            # 只读取这一列数据
            data_gctx = parse(self.gctx_file, cid=[sig_id])
            gene_expression = data_gctx.data_df.values.flatten()
        except Exception as e:
            # 偶尔会有 sig_id 在 gctx 里找不到的情况，跳过即可
            # print(f"Error reading {sig_id}: {e}")
            return None

        # 5. 打包返回 (一共 4 样东西)
        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(adj, dtype=torch.float32),
            torch.tensor(fp_array, dtype=torch.float32),  # 新增
            torch.tensor(gene_expression, dtype=torch.float32)
        )