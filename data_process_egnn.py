import pickle
import json
import torch
import os
import numpy as np
import scipy.sparse as sp
import networkx as nx

from torch_geometric import data as DATA
from collections import OrderedDict
from rdkit import Chem
from utils_egnn import DTADataset

from Bio.PDB import PDBParser
from rdkit.Chem import AllChem
import math
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}
res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}
res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}
res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}
res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}
res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}
res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def load_data(dataset):
    affinity = pickle.load(open('./source/data/' + dataset + '/affinities', 'rb'), encoding='latin1')
    if dataset == 'davis':
        affinity = -np.log10(affinity / 1e9)
    return affinity


# 【严格匹配主程序的接口】只返回数据集！
def process_data(affinity_mat, dataset, scenario='warm'):
    import numpy as np
    import json
    dataset_path = './source/data/' + dataset + '/'
    rows, cols = np.where(np.isnan(affinity_mat) == False)

    print(f"========== 当前正在构建 {scenario} 场景的数据集 ==========")

    if scenario == 'warm':
        train_file = json.load(open(dataset_path + 'train_set.txt'))
        train_index = [idx for fold in train_file for idx in fold]
        test_index = json.load(open(dataset_path + 'test_set.txt'))

        train_edges_idx = [idx for idx in train_index if idx < len(rows)]
        test_edges_idx = [idx for idx in test_index if idx < len(rows)]

    elif scenario == 'S1':
        train_folds = json.load(open(dataset_path + 'S1_train_set.txt'))
        train_drugs = [idx for fold in train_folds for idx in fold]
        test_drugs = json.load(open(dataset_path + 'S1_test_set.txt'))

        train_edges_idx = [i for i in range(len(rows)) if rows[i] in train_drugs]
        test_edges_idx = [i for i in range(len(rows)) if rows[i] in test_drugs]

    elif scenario == 'S2':
        train_folds = json.load(open(dataset_path + 'S2_train_set.txt'))
        train_targets = [idx for fold in train_folds for idx in fold]
        test_targets = json.load(open(dataset_path + 'S2_test_set.txt'))

        train_edges_idx = [i for i in range(len(cols)) if cols[i] in train_targets]
        test_edges_idx = [i for i in range(len(cols)) if cols[i] in test_targets]

    elif scenario == 'S3':
        train_drug_folds = json.load(open(dataset_path + 'S1_train_set.txt'))
        train_drugs = [idx for fold in train_drug_folds for idx in fold]
        test_drugs = json.load(open(dataset_path + 'S1_test_set.txt'))

        train_target_folds = json.load(open(dataset_path + 'S2_train_set.txt'))
        train_targets = [idx for fold in train_target_folds for idx in fold]
        test_targets = json.load(open(dataset_path + 'S2_test_set.txt'))

        train_edges_idx = [i for i in range(len(rows)) if rows[i] in train_drugs and cols[i] in train_targets]
        test_edges_idx = [i for i in range(len(rows)) if rows[i] in test_drugs and cols[i] in test_targets]

    else:
        raise ValueError("不支持的 scenario 类型")

    train_rows, train_cols = rows[train_edges_idx], cols[train_edges_idx]
    train_Y = affinity_mat[train_rows, train_cols]
    train_dataset = DTADataset(drug_ids=train_rows, target_ids=train_cols, y=train_Y)

    test_rows, test_cols = rows[test_edges_idx], cols[test_edges_idx]
    test_Y = affinity_mat[test_rows, test_cols]
    test_dataset = DTADataset(drug_ids=test_rows, target_ids=test_cols, y=test_Y)

    return train_dataset, test_dataset


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

def get_drug_molecule_graph(ligands, dataset):
    smile_graph = OrderedDict()
    ori_78_dir = f'./new_train/drug_graphs/{dataset}/'
    import os
    os.makedirs(ori_78_dir, exist_ok=True)
    for d in ligands.keys():
        save_path = ori_78_dir + d + '.npy'
        if not os.path.exists(save_path):
            lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
            g = smile_to_graph(lg, dataset, d)
            np.save(save_path, np.array(g, dtype=object), allow_pickle=True)
            print(f"✅ 成功生成并保存药物图: {save_path}")
        smile_graph[d] = np.load(save_path, allow_pickle=True)
    return smile_graph

def generate_3d_coordinates(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无法从SMILES字符串创建分子")
    AllChem.EmbedMolecule(mol, randomSeed=42)
    num_conformers_before = mol.GetNumConformers()
    if num_conformers_before == 0:
        AllChem.EmbedMultipleConfs(mol, numConfs=1, maxAttempts=1000, useRandomCoords=True, randomSeed=42)
        num_conformers_before = mol.GetNumConformers()
        if num_conformers_before == 0:
            raise ValueError("嵌入后分子没有构象")
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        AllChem.UFFOptimizeMolecule(mol)
    conformer = mol.GetConformer()
    coordinates = np.array([conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=np.float32)
    return np.array(coordinates)

def get_drug_edgeweight(mol):
    edge_features = []
    for bond in mol.GetBonds():
        bt = bond.GetBondType()
        bond_features = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bond.IsInRing()
        ]
        edge_features.append(bond_features)
    return np.array(edge_features)

def chemformer_embed(id,dataset):
    embed_dir = f'./source/data/{dataset}/drug_embed/chemformer/'
    embed_file = embed_dir + id + '.npy'
    features = np.load(embed_file)
    return features

def smile_to_graph(smile,dataset,id):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))
    features = np.array(features)
    edge_index = []
    for bond in mol.GetBonds():
        edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    coordinates = generate_3d_coordinates(smile)
    edge_weight = get_drug_edgeweight(mol)
    return c_size, features, edge_index,coordinates,edge_weight

def get_target_molecule_graph(proteins, dataset):
    msa_path = './source/data/' + dataset + '/aln'
    contac_path = './source/data/' + dataset + '/pconsc4'
    target_graph = OrderedDict()
    save_dir = f'./new_train/protein_graphs/{dataset}/'
    import os
    os.makedirs(save_dir, exist_ok=True)
    for t in proteins.keys():
        save_path = save_dir + t + '.npy'
        if not os.path.exists(save_path):
            g = target_to_graph(t, proteins[t], contac_path, msa_path, dataset)
            np.save(save_path, np.array(g, dtype=object), allow_pickle=True)
            print(f"✅ 成功生成并保存蛋白质图: {save_path}")
        g = np.load(save_path, allow_pickle=True)
        size = g[0]
        features = g[1]
        edge_index = g[2]
        coords = g[3]
        edge_weight = g[4]
        target_graph[t] = size, features, edge_index, coords, edge_weight
    return target_graph

def get_protein_coordinates(pdb_file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file_path)
    model = structure[0]
    coordinates = []
    for chain in model:
        for residue in chain.get_residues():
            if 'CA' in residue:
                atom = residue['CA']
                coordinates.append(atom.coord)
    return np.array(coordinates, dtype=np.float32)

def cos_sim(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def cal_angle(point_a, point_b, point_c):
    a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]
    a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]
    if len(point_a) == len(point_b) == len(point_c) == 3:
        a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]
    else:
        a_z, b_z, c_z = 0, 0, 0
    x1, y1, z1 = (a_x-b_x), (a_y-b_y), (a_z-b_z)
    x2, y2, z2 = (c_x-b_x), (c_y-b_y), (c_z-b_z)
    cos_b = (x1*x2 + y1*y2 + z1*z2) / (math.sqrt(x1**2 + y1**2 + z1**2) * (math.sqrt(x2**2 + y2**2 + z2**2)))
    return cos_b

def get_target_edgeweight_optimized(contact_map, ca_coords, target_feature):
    edge_features = []
    target_edge_index = []
    for i in range(len(contact_map)):
        for j in range(len(contact_map)):
            contact_ij = contact_map[i][j]
            if i != j and contact_ij >= 0.5:
                target_edge_index.append([i, j])
                sim_ij = cos_sim(target_feature[i], target_feature[j])
                if contact_map[i][j] <= 0.5:
                    dis_ij = 0.5
                else:
                    dis_ij = 1 / contact_map[i][j]
                angle_ij = cal_angle(ca_coords[i], [0, 0, 0], ca_coords[j])
                contact_features_ij = [sim_ij, dis_ij, angle_ij]
                edge_features.append(contact_features_ij)
    return edge_features,target_edge_index

def get_ESM2_embed(dataset,id):
    embed_dir = f'./source/data/{dataset}/pro_embed/ESM2-33dim/'
    load_file = embed_dir + id + '.npy'
    esm2_embed = np.load(load_file,allow_pickle=True)
    return esm2_embed

def target_to_graph(target_key, target_sequence, contact_dir, aln_dir, dataset):
    import os
    import numpy as np
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    pdb_path = f'./source/data/{dataset}/PDB/{target_key}.pdb'
    ca_coords = get_protein_coordinates(pdb_path)
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    node_feature = target_feature
    min_len = min(len(ca_coords), len(node_feature), contact_map.shape[0])
    ca_coords = ca_coords[:min_len]
    node_feature = node_feature[:min_len]
    contact_map = contact_map[:min_len, :min_len]
    edge_weight, target_edge_index = get_target_edgeweight_optimized(contact_map, ca_coords, node_feature)
    size = min_len
    return size, node_feature, target_edge_index, ca_coords, edge_weight

def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)

def target_to_feature(target_key, target_sequence, aln_dir):
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    feature = target_feature(aln_file, target_sequence)
    return feature

def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    return pssm_mat

def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)

def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        pro_hot[i, ] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i, ] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)