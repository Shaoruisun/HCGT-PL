# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import csv
import time
import pandas as pd
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.autograd import Variable

import math
import numpy as np
import torch.optim as optim
import torch.nn as nn
#from torch_geometric.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
import argparse

from data_entry import select_train_loader, select_eval_loader
from model_entry import select_model
from options import prepare_train_args
from logger import Logger
from torch_utils import load_match_dict
from dgl.data.utils import load_graphs
from metrics import *
from list_dataset import *

criterion = nn.BCEWithLogitsLoss()
class Trainer:
    def __init__(self):
        args = prepare_train_args()
        self.args = args
        torch.manual_seed(args.seed)#为CPU设置种子用于生成随机数，以使得结果是确定的
        self.logger = Logger(args)
        print(self.args)
        self.train_loader = select_train_loader(args)
        self.val_loader = select_eval_loader(args)

        #self.model = select_model(args)
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.model=select_model(args)
        if args.load_model_path != '':
            if args.load_not_strict:
                #load_match_dict(self.model, args.load_model_path)
                pass
            else:
                print('调用预训练')
                self.model.load_state_dict(torch.load(args.load_model_path))#.state_dict())

        #self.model = torch.nn.DataParallel(self.model)分布式训练
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.args.lr
                                          , betas=(self.args.momentum, self.args.beta),
                                          weight_decay=self.args.weight_decay
                                         )

    def train(self):
        loss_list=[]
        personR_list=[]
        for epoch in range(self.args.epochs):
            train_loss=self.train_per_epoch(epoch)
            test_loss=self.val_per_epoch(epoch)
            self.logger.save_curves(epoch)
            self.logger.save_check_point(self.model,epoch)#保存模型
            print('epoch {:d} | train loss {:.4f} | val loss {:.4f}'.format(epoch, train_loss, test_loss))
            #print('PersonR {:.4f}'.format(personR[0]))
            loss_list.append([float(train_loss), float(test_loss)])
            #personR_list.append(personR)
        train_loss=pd.DataFrame(data=loss_list)#数据有三列，列名分别为one,two,three
        train_loss.to_csv('loss_list.csv',encoding='gbk')
        #personR_list_csv=pd.DataFrame(data=personR_list)#数据有三列，列名分别为one,two,three
        #personR_list_csv.to_csv('personR_list.csv',encoding='gbk')
        #eval_loss=pd.DataFrame(data=eval_loss)#数据有三列，列名分别为one,two,three
        #train_loss.to_csv('eval_loss.csv',encoding='gbk')
        
        
            

    def train_per_epoch(self, epoch,aux_weight=0.001):
        # switch to train mode
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
        #device = torch.device('cuda:0')
        self.model.train()
     
        for i, data in enumerate(self.train_loader):
            self.optimizer.zero_grad()
             
            pdbids,bg_0,bg_1,label= data
            bg_0=bg_0.to(device)
            bg_1=bg_1.to(device)
            self.model.to(device)
            label=label.to(device)
            output_2,output_3,output_0 ,output_1,regression = self.model(bg_0,bg_1)#.to(device)

            label=label.t()

            loss =criterion(regression.view(-1), label)
            loss.backward()
            self.optimizer.step()
            return loss

    def val_per_epoch(self, epoch):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 单GPU或者CPU
        self.model.eval()
        for i, data in enumerate(self.val_loader):

            pdbids,bg_0,bg_1,label= data
            bg_0=bg_0.to(device)
            bg_1=bg_1.to(device)
            self.model.to(device)
            label=label.to(device)
            label=label.t()
            with torch.no_grad():
                output_2,output_3,output_0 ,output_1,regression= self.model(bg_0,bg_1)#.to(device)
            loss =criterion(regression.view(-1), label)
     
        return loss    

def main():
    trainer = Trainer()
    trainer.train()


if __name__ == '__main__':
    main()
    