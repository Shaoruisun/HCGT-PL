import torch.nn as nn
import torch
from fcn import DTI

#device = torch.device('cuda:0')
def select_model(args):
    type2model = {
        'DTI': DTI(node_feat_size=35, edge_feat_size=17, hidden_feat_size=128)
    }
    model = type2model[args.model_type]
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model