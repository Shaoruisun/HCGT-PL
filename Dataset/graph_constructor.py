# %%
import os
import pickle
from glob import glob
import numpy as np
import argparse
import re
import torch 
import dgl
import random
from torch.utils.data import DataLoader

from scipy.spatial import distance_matrix
import networkx as nx
import multiprocessing
from itertools import repeat
from collections import defaultdict


import prolif as plf
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from utils import *
import warnings
warnings.filterwarnings('ignore')

# %%

def one_of_k_encoding(k, possible_values):
    if k not in possible_values:
        raise ValueError(f"{k} is not a valid value in {possible_values}")
    return [k == e for e in possible_values]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):

    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])

        atom_feats = np.array(results).astype(np.float32)

        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(atom_feats))

def edge_features(mol, graph):
    geom = mol.GetConformers()[0].GetPositions()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for neighbor in mol.GetAtomWithIdx(j).GetNeighbors():
            k = neighbor.GetIdx() 
            if mol.GetBondBetweenAtoms(j, k) is not None and i != k:
                # geometrical features
                vector1 = geom[j] - geom[i]
                vector2 = geom[k] - geom[i]
                # angle between two vectors
                angles_ijk.append(angle(vector1, vector2))
                # area enclosed by two vectors
                areas_ijk.append(area_triangle(vector1, vector2))
                # distance between two vectors
                dists_ik.append(cal_dist(geom[i], geom[k]))

        angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
        areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
        dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
        dist_ij1 = cal_dist(geom[i], geom[j], ord=1)
        dist_ij2 = cal_dist(geom[i], geom[j], ord=2)
        # length = 11
        geom_feats = [
            angles_ijk.max()*0.1,
            angles_ijk.sum()*0.01,
            angles_ijk.mean()*0.1,
            areas_ijk.max()*0.1,
            areas_ijk.sum()*0.01,
            areas_ijk.mean()*0.1,
            dists_ik.max()*0.1,
            dists_ik.sum()*0.01,
            dists_ik.mean()*0.1,
            dist_ij1*0.1,
            dist_ij2*0.1,
        ]

        bond_type = bond.GetBondType()
        basic_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]

        graph.add_edge(i, j, feats=torch.tensor(basic_feats+geom_feats).float())

def mol_with_atom_index(mol):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol
def get_nonBond_pair(m1,m2):
    """
    docstring: 
        calculate IFP(interaction fingerprint)
    input:
        m1: rdkit.Chem.Mol(ligand)
        m2: rdkit.Chem.Mol(protein)
    output:
        nonbond_pairs: list of tuple (ligand_atom_idx,protein_atom_idx)
        inter_types: list of interaction type
    """
    fp = plf.Fingerprint()
    prot = plf.Molecule(m2)
    ligand = plf.Molecule.from_rdkit(m1)
    fp.run_from_iterable([ligand],prot,progress=False,n_jobs = 1)
    df = fp.to_dataframe(return_atoms=True)
    # find IFP  atom pairs , match with rdkit id 
    res_to_idx = defaultdict(dict)
    for atom in m2.GetAtoms():
        atom_idx = atom.GetIdx()
        res_name = str(plf.residue.ResidueId.from_atom(atom))
        res_to_idx[res_name][len(res_to_idx[res_name])] = atom_idx

    nonbond_pairs = []
    inter_types = []
    for key in df.keys():
        ligand_name,res_name,inter_type = key
        lig_atom_num,res_atom_num = df[key][0]
        pdb_atom_idx = res_to_idx[res_name][res_atom_num]
        nonbond_pairs.append((lig_atom_num,pdb_atom_idx))  
        inter_types.append(inter_type) 

    return nonbond_pairs,inter_types        
def get_one_hot_for_atoms(atom_index_1, atom_index_2,inter_types,nonbond_pairs):
    interaction_types = ['Hydrophobic', 'PiCation', 'CationPi', 'PiStacking']  
    interactions_dict = {}  
    for (atom1, atom2), interaction in zip(nonbond_pairs, inter_types):  
        pair = (atom1, atom2) if atom1 < atom2 else (atom2, atom1)  
        if pair not in interactions_dict:  
            interactions_dict[pair] = []  
        interactions_dict[pair].append(interaction)  
    pair = (atom_index_1, atom_index_2) if atom_index_1 < atom_index_2 else (atom_index_2, atom_index_1)  
    one_hot = np.zeros(len(interaction_types), dtype=int)     
    if pair in interactions_dict:  
        interactions = interactions_dict[pair]  
        for interaction in interactions:  
            if interaction in interaction_types:  
                index = interaction_types.index(interaction)  
                one_hot[index] = 1  
    return one_hot        

def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    edge_features(mol, graph)

    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    
    if graph.number_of_edges() > 0:
        edge_index = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph.edges(data=True)]).T
        edge_attr = torch.stack([feats['feats'] for u, v, feats in graph.edges(data=True)])
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 17), dtype=torch.float)  # 17 = 6 basic features + 11 geometric features

    return x, edge_index, edge_attr

def geom_feat(pos_i, pos_j, pos_k, angles_ijk, areas_ijk, dists_ik):
    vector1 = pos_j - pos_i
    vector2 = pos_k - pos_i
    angles_ijk.append(angle(vector1, vector2))
    areas_ijk.append(area_triangle(vector1, vector2))
    dists_ik.append(cal_dist(pos_i, pos_k))

def geom_feats(pos_i, pos_j, angles_ijk, areas_ijk, dists_ik):
    angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
    areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
    dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
    dist_ij1 = cal_dist(pos_i, pos_j, ord=1)
    dist_ij2 = cal_dist(pos_i, pos_j, ord=2)
    # length = 11
    geom = [
        angles_ijk.max()*0.1,
        angles_ijk.sum()*0.01,
        angles_ijk.mean()*0.1,
        areas_ijk.max()*0.1,
        areas_ijk.sum()*0.01,
        areas_ijk.mean()*0.1,
        dists_ik.max()*0.1,
        dists_ik.sum()*0.01,
        dists_ik.mean()*0.1,
        dist_ij1*0.1,
        dist_ij2*0.1,
    ]
    return geom

def inter_graph(ligand, pocket, dis_threshold =5):
    graph_l2p = nx.DiGraph()
    graph_p2l = nx.DiGraph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix<dis_threshold)
    
    #nonbond_pairs,inter_types=get_nonBond_pair(ligand,pocket)
    #'Hydrophobic', 'HBDonor', 'HBAcceptor', 'Cationic', 'Anionic', 'PiCation', 'CationPi', 'PiStacking'
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()
    for i, j in zip(node_idx[0], node_idx[1]):
        # ligand to pocket
        ks = node_idx[0][node_idx[1] == j]
        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for k in ks:
            if k != i:
                geom_feat(pos_l[i], pos_p[j], pos_l[k], angles_ijk, areas_ijk, dists_ik)
        geom = geom_feats(pos_l[i], pos_p[j], angles_ijk, areas_ijk, dists_ik)
        #print(i,j)
        #print(nonbond_pairs)
        #if (i,j) in nonbond_pairs:
            #print(i,j)
        #    one_hot=get_one_hot_for_atoms(i,j,inter_types,nonbond_pairs)
        #else:
        #    one_hot=[0, 0, 0, 0]
        #bond_feats=torch.cat((torch.FloatTensor(geom), torch.FloatTensor(one_hot)), dim=0) 
        bond_feats = torch.FloatTensor(geom)
        graph_l2p.add_edge(i, j, feats=bond_feats)

        # pocket to ligand
        ks = node_idx[1][node_idx[0] == i]
        angles_ijk = []
        areas_ijk = []
        dists_ik = []
        for k in ks:
            if k != j:
                geom_feat(pos_p[j], pos_l[i], pos_p[k], angles_ijk, areas_ijk, dists_ik)     
        geom = geom_feats(pos_p[j], pos_l[i], angles_ijk, areas_ijk, dists_ik)
        #if (i,j) in nonbond_pairs:
        #    one_hot=get_one_hot_for_atoms(i,j,inter_types,nonbond_pairs)
        #else:
        #    one_hot=[0, 0, 0, 0]
        #bond_feats=torch.cat((torch.FloatTensor(geom), torch.FloatTensor(one_hot)), dim=0)
        bond_feats = torch.FloatTensor(geom)
        graph_p2l.add_edge(j, i, feats=bond_feats)
    
    if graph_l2p.number_of_edges() > 0:
        edge_index_l2p = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph_l2p.edges(data=True)]).T
        edge_attr_l2p = torch.stack([feats['feats'] for u, v, feats in graph_l2p.edges(data=True)])
    else:
        edge_index_l2p = torch.empty((2, 0), dtype=torch.long)
        edge_attr_l2p = torch.empty((0, 11), dtype=torch.float)

    if graph_p2l.number_of_edges() > 0:
        edge_index_p2l = torch.stack([torch.LongTensor((u, v)) for u, v, feats in graph_p2l.edges(data=True)]).T
        edge_attr_p2l = torch.stack([feats['feats'] for u, v, feats in graph_p2l.edges(data=True)])
    else:
        edge_index_p2l = torch.empty((2, 0), dtype=torch.long)
        edge_attr_p2l = torch.empty((0, 11), dtype=torch.float)

    return (edge_index_l2p, edge_attr_l2p), (edge_index_p2l, edge_attr_p2l)

def load_mol(molpath, explicit_H, use_chirality):
	# load mol
	if re.search(r'.pdb$', molpath):
		mol = Chem.MolFromPDBFile(molpath, removeHs=not explicit_H)#, sanitize=False)
	elif re.search(r'.mol2$', molpath):
		mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H)
	elif re.search(r'.sdf$', molpath):			
		mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H)
	else:
		raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")	
	
	if use_chirality:
		Chem.AssignStereochemistryFrom3D(mol)
	return mol

# %%
def mols2graphs(pocket_path, ligand_path,pdbid, dis_threshold=5.0):
    #try:
    pocket = load_mol(pocket_path, explicit_H=False,use_chirality=False) 
    ligand = load_mol(ligand_path, explicit_H=False,use_chirality=False)
    #print(ligand)
    # the distance threshold to determine the interaction between ligand atoms and protein atoms
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    x_l, edge_index_l, edge_attr_l = mol2graph(ligand)
    x_p, edge_index_p, edge_attr_p = mol2graph(pocket)
    (edge_index_l2p, edge_attr_l2p), (edge_index_p2l, edge_attr_p2l) = inter_graph(ligand, pocket, dis_threshold=dis_threshold)

    graph_data = {
        ('ligand', 'intra_l', 'ligand') : (edge_index_l[0], edge_index_l[1]),
        ('pocket', 'intra_p', 'pocket') : (edge_index_p[0], edge_index_p[1]),
        ('ligand', 'inter_l2p', 'pocket') : (edge_index_l2p[0], edge_index_l2p[1]),
        ('pocket', 'inter_p2l', 'ligand') : (edge_index_p2l[0], edge_index_p2l[1])
    }
    g = dgl.heterograph(graph_data, num_nodes_dict={"ligand":atom_num_l, "pocket":atom_num_p})
    g.nodes['ligand'].data['h'] = x_l
    g.nodes['pocket'].data['h'] = x_p
    g.edges['intra_l'].data['e'] = edge_attr_l
    g.edges['intra_p'].data['e'] = edge_attr_p
    g.edges['inter_l2p'].data['e'] = edge_attr_l2p
    g.edges['inter_p2l'].data['e'] = edge_attr_p2l
    return pdbid,g
    #except:
    #    g = None

    
def drop_nodes(data, ratio):
    """
    使用dgl.DGLGraph删除给定数据中的节点，并相应地更新边索引。

    参数:
        data (dgl.DGLGraph): 包含节点和边索引的图数据对象
        ratio (float): 要删除的节点比例

    返回:
        data (dgl.DGLGraph): 更新后的图数据对象
    """
    #print(data.nodes)
    # 获取节点数量
    node_num = data.number_of_nodes()

    # 计算需要删除的节点数量
    drop_num = int(node_num * ratio)

    # 从节点中随机选择需要删除的节点的索引
    idx_drop = np.random.choice(node_num, drop_num, replace=False)

    # 删除节点及其关联的边
    data.remove_nodes(idx_drop)

    # 返回更新后的图数据对象
    return data


def permute_edges(data, ratio):
    """
    使用dgl.DGLGraph对给定数据进行边置换操作，并更新边索引。

    参数:
        data (dgl.DGLGraph): 包含节点和边索引的图数据对象
        ratio (float): 边置换的比例

    返回:
        data (dgl.DGLGraph): 更新后的图数据对象
    """

    # 获取边数量
    edge_num = data.number_of_edges()

    # 计算需要置换的边数量
    permute_num = int(edge_num * ratio)

    # 从所有边中随机选择需要保留的边的索引
    selected_edges = np.random.choice(edge_num, permute_num, replace=False)

    # 删除未选中的边
    data.remove_edges(selected_edges)

    # 返回更新后的图数据对象
    return data



def subgraph(data, ratio):
    """
    使用dgl.DGLGraph对给定数据进行子图提取操作，并更新边索引。

    参数:
        data (dgl.DGLGraph): 包含节点和边索引的图数据对象
        ratio (float): 子图中节点的比例

    返回:
        data (dgl.DGLGraph): 更新后的图数据对象
    """

    # 获取节点数量
    node_num = data.number_of_nodes()

    # 计算子图中需要保留的节点数量
    sub_num = int(node_num * ratio)

    # 从节点中随机选择一个起始节点
    idx_sub = [np.random.randint(0, node_num, size=(1,)).item()]

    # 获取与起始节点相邻的节点索引
    idx_neigh = set(data.successors(idx_sub[0]))

    count = 0
    while len(idx_sub) <= sub_num:
        count = count + 1
        if count > node_num:
            break
        if len(idx_neigh) == 0:
            break

        # 从相邻节点中随机选择一个节点
        sample_node = np.random.choice(list(idx_neigh)).item()

        # 如果已选择的节点已在子图中，则继续选择下一个节点
        if sample_node in idx_sub:
            continue

        # 将选择的节点添加到子图中
        idx_sub.append(sample_node)

        # 更新相邻节点集合
        idx_neigh.update(data.successors(sample_node))

    # 根据子图节点和非子图节点生成索引字典
    idx_drop = [n for n in range(node_num) if not n in idx_sub]
    # 通过保留子图节点更新图的边索引
    data = data.subgraph(idx_drop)

    # 返回更新后的图数据对象
    return data
