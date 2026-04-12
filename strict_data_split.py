import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import warnings

warnings.filterwarnings('ignore')


def create_drug_cell_line_split():
    print("🚀 启动 [Drug-Cell Line Split] 数据切分引擎...")

    # 1. 读取原始数据
    csv_path = './data/clean_dataset.csv'
    metrics_path = './data/GSE92742_Broad_LINCS_sig_metrics.txt'

    df = pd.read_csv(csv_path)
    metrics_df = pd.read_csv(metrics_path, sep='\t')

    # 2. 清洗数据 (过滤 TAS < 0.2 和 restricted)
    print("🧹 正在清洗数据 (过滤 TAS < 0.2 和 restricted)...")
    df = df.merge(metrics_df[['sig_id', 'tas']], on='sig_id', how='left')
    df = df[df['tas'] >= 0.2]

    # 🔥 修复 1：把大写 SMILES 改成了小写 smiles
    if 'smiles' in df.columns:
        df = df[df['smiles'] != 'restricted']
    else:
        print("❌ 找不到 smiles 列，请检查你的 CSV 表头！")
        return

    df = df.reset_index(drop=True)
    print(f"📦 清洗完毕！获得完美数据大水池: {len(df)} 条")

    # 3. 核心修复：从 sig_id 里面自动抠出细胞系！
    # 例子: 'ABY001_A549_XH:BRD-K81418486:10:3' -> 按照 '_' 劈开 -> 拿到 'A549'
    print("🧬 正在从 sig_id 中智能提取细胞系特征...")
    df['cell_id'] = df['sig_id'].apply(lambda x: x.split('_')[1] if '_' in str(x) else 'UNKNOWN')

    # 组合 "药物+细胞系" 对
    df['drug_cell_pair'] = df['smiles'] + "_" + df['cell_id']

    # 4. 按照 "药物-细胞系对" 进行严谨隔离切分 (85% 训练, 15% 测试)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['drug_cell_pair']))

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    # --- 严格审查 ---
    train_pairs = set(train_df['drug_cell_pair'])
    test_pairs = set(test_df['drug_cell_pair'])
    overlap = train_pairs.intersection(test_pairs)

    print("\n" + "=" * 50)
    print("🛡️ 【学术诚信审查：Drug-Cell Line Split 报告】")
    print(f"📊 训练集: {len(train_df)} 条 (包含 {len(train_pairs)} 种独特药-细胞组合)")
    print(f"📊 测试集: {len(test_df)} 条 (包含 {len(test_pairs)} 种独特药-细胞组合)")
    print(f"🚨 重叠组合数量 (Data Leakage): {len(overlap)} 种")

    if len(overlap) == 0:
        print("✅ 审查通过！0 泄露！完美契合你的 Pan-Cancer Repurposing 论文主旨！")

    # 5. 保存结果！我们只保留需要的列，让文件变小变清爽
    columns_to_save = ['sig_id', 'smiles', 'drug_id', 'cell_id']
    train_df[columns_to_save].to_csv('./data/train_dataset_drugcell.csv', index=False)
    test_df[columns_to_save].to_csv('./data/test_dataset_drugcell.csv', index=False)
    print("\n💾 训练集和测试集已保存 (train_dataset_drugcell.csv / test_dataset_drugcell.csv)！")
    print("=" * 50)


if __name__ == '__main__':
    create_drug_cell_line_split()