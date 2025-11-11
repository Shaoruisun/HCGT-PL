import torch as th
import dgl
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset  # , DataLoader
import pandas as pd
import numpy as np
from rdkit import Chem
from joblib import Parallel, delayed
import os
import tempfile
import shutil
class PDBbindDataset(Dataset):
    def __init__(self,
                 ids=None,
                 protein_ligand_1=None,
                 target=None
                 ):
        if isinstance(ids, np.ndarray) or isinstance(ids, list):
            self.pdbids = ids
        else:
            try:
                self.pdbids = np.load(ids)
            except:
                raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
        if isinstance(protein_ligand_1, np.ndarray) or isinstance(protein_ligand_1, tuple) or isinstance(protein_ligand_1, list):
            if isinstance(protein_ligand_1[0], dgl.DGLGraph):
                self.graphs_1 = protein_ligand_1
            else:
                raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')
        else:
            try:
                self.graphs_1, _ = load_graphs(protein_ligand_1)
            except:
                raise ValueError('the variable "ligs" should be a set of (or a file to store) dgl.DGLGraph objects.')
        if isinstance(target, np.ndarray) or isinstance(target, list):
            self.label = target
        else:
            try:
                self.label = np.load(target)
            except:
                raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')  
    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.pdbids[idx],self.graphs_1[idx],self.graphs_1[idx],self.label[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.pdbids)

    def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
        # random.seed(seed)
        np.random.seed(seed)
        if valnum is None:
            valnum = int(valfrac * len(self.pdbids))
        val_inds = np.random.choice(np.arange(len(self.pdbids)), valnum, replace=False)
        train_inds = np.setdiff1d(np.arange(len(self.pdbids)), val_inds)
        return train_inds, val_inds
