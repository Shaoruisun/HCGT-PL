from list_dataset import PDBbindDataset
#from list_dataset import PDBbindDataset
from torch.utils.data import DataLoader
import dgl
import numpy as np
import torch
from options import prepare_train_args
import argparse
args=prepare_train_args()
data_dir='/Pre-train'
data=PDBbindDataset(ids="%s/out_id_pre_train.npy"%(data_dir),protein_ligand_1="%s/out_pre_train_1.bin"%(data_dir),protein_ligand_2="%s/out_pre_train_2.bin"%(data_dir))
train_inds, val_inds =data.train_and_test_split(valnum=20000, seed=10)#2023#1000
print('训练集个数',len(train_inds))
print('测试集个数',len(val_inds))

def collate(data):
	pdbids, graphs_0, graphs_1,= map(list, zip(*data))
	bg_0 = dgl.batch(graphs_0)
	bg_1 = dgl.batch(graphs_1)
	for nty in bg_0.ntypes:
		bg_0.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
	for ety in bg_0.canonical_etypes:
		bg_0.set_e_initializer(dgl.init.zero_initializer, etype=ety)
	for nty in bg_1.ntypes:
		bg_1.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
	for ety in bg_1.canonical_etypes:
		bg_1.set_e_initializer(dgl.init.zero_initializer, etype=ety)

        
	return pdbids,bg_0,bg_1




def get_dataset_by_type_train(args, is_train=False):
    type2data = {
        'PDBbind2016': PDBbindDataset(ids=data.pdbids[train_inds],
							protein_ligand_1=np.array(data.graphs_1)[train_inds],
							protein_ligand_2=np.array(data.graphs_2)[train_inds])}
    dataset = type2data['PDBbind2016']
    return dataset
def get_dataset_by_type_val(args, is_train=False):
    type2data = {
        'PDBbind2016': PDBbindDataset(ids=data.pdbids[val_inds],
							protein_ligand_1=np.array(data.graphs_1)[val_inds],
							protein_ligand_2=np.array(data.graphs_2)[val_inds])
    }
    dataset = type2data['PDBbind2016']
    return dataset


def select_train_loader(args):
    # usually we need loader in training, and dataset in eval/test
    train_dataset = get_dataset_by_type_train(args, True)
    print('{} samples found in train'.format(len(train_dataset)))
    train_loader = DataLoader(train_dataset,args.batch_size, shuffle=True, num_workers=8,collate_fn=collate,drop_last=False)#, pin_memory=True, drop_last=False)
    return train_loader

#args.batch_size
def select_eval_loader(args):
    eval_dataset = get_dataset_by_type_val(args)
    print('{} samples found in val'.format(len(eval_dataset)))
    val_loader = DataLoader(eval_dataset,args.batch_size, shuffle=False, num_workers=8,collate_fn=collate,drop_last=False)#, pin_memory=True, drop_last=False)
    return val_loader

