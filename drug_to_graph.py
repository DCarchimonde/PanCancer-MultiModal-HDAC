import numpy as np
from rdkit import Chem


def get_atom_features(atom):
    """
    Extract one-hot encoded features for a given atom based on its chemical symbol.
    """
    permitted_list_of_atoms = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I',
        'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge',
        'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    ]

    atom_type = atom.GetSymbol()
    if atom_type not in permitted_list_of_atoms:
        atom_type = 'Unknown'

    atom_features = [int(atom_type == x) for x in permitted_list_of_atoms]
    return np.array(atom_features)


def smiles_to_graph(smiles):
    """
    Convert a SMILES string into a graph representation suitable for GNNs.

    Returns:
        features (np.ndarray): Node feature matrix of shape (num_atoms, num_features).
        adj (np.ndarray): Adjacency matrix of shape (num_atoms, num_atoms).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Construct node feature matrix (X)
    features = []
    for atom in mol.GetAtoms():
        features.append(get_atom_features(atom))
    features = np.array(features)

    # Construct adjacency matrix (A)
    adj = Chem.GetAdjacencyMatrix(mol)

    return features, adj


if __name__ == "__main__":
    # Internal module testing
    sample_smiles = "CCNC(=O)CCC(N)C(O)=O"
    print(f"Processing testing SMILES: {sample_smiles}")

    x, a = smiles_to_graph(sample_smiles)

    if x is not None:
        print(f"Node Feature Matrix (X) Shape: {x.shape}")
        print(f"Adjacency Matrix (A) Shape: {a.shape}")
        print("Graph extraction successful. Ready for model ingestion.")
    else:
        print("Failed to parse SMILES sequence.")