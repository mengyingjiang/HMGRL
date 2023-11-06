from rdkit import Chem
import networkx as nx

def process(file_name):
    id_graph_dict = dict()
    with open(file_name) as f:
        for line in f:
            line = line.rstrip()
            index, id, target, enzyme, smiles, name, smiles1 = line.split()
            mol = Chem.MolFromSmiles(smiles1)
            graph = mol_to_nx(mol)
            id_graph_dict[id] = graph
    # with open('/data/hancwang/DDI/code/data/id_graph_dict.json', 'w') as f:
    #     json.dump(id_graph_dict, f)
    return id_graph_dict

def mol_to_nx(mol):
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   symbol=atom.GetSymbol(),
                   formal_charge=atom.GetFormalCharge(),
                   implicit_valence=atom.GetImplicitValence(),
                   ring_atom=atom.IsInRing(),
                   degree=atom.GetDegree(),
                   hybridization=atom.GetHybridization())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType())
    return G