from rdkit import Chem
from rdkit.Chem import Lipinski,Descriptors

def calculate_aromatic_proportion(mol):
    # Convert SMILES to a molecule object
    
    if mol is None:
        return 0
    
    
    aromatic_atoms = [atom for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    num_aromatic_atoms = len(aromatic_atoms)
    
 
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    
    
    if num_heavy_atoms == 0:
        return 0
        
    aromatic_proportion = num_aromatic_atoms / num_heavy_atoms
    return aromatic_proportion

def get_molecular_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    features = {
        'MolWt': Descriptors.MolWt(mol),
        'MolLogP': Descriptors.MolLogP(mol),
        'MolMR': Descriptors.MolMR(mol),
        'HeavyAtomCount': Lipinski.HeavyAtomCount(mol),
        'NumHAcceptors': Lipinski.NumHAcceptors(mol),
        'NumHDonors': Lipinski.NumHDonors(mol),
        'NumHeteroatoms': Lipinski.NumHeteroatoms(mol),
        'NumRotatableBonds': Lipinski.NumRotatableBonds(mol),
        'NumValenceElectrons': Descriptors.NumValenceElectrons(mol),
        'NumAromaticRings': Lipinski.NumAromaticRings(mol),
        'RingCount': Lipinski.RingCount(mol),
        'TPSA': Descriptors.TPSA(mol),
        'LabuteASA': Descriptors.LabuteASA(mol),
        'aromatic propotion': calculate_aromatic_proportion(mol) 
    }


    return features