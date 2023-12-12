import os
import torch
import argparse
from typing import Tuple
from utils.file_utils import pickle_file_load
from utils.trainer_utils import (
    bce_calculation,
    ce_calculation
)
import numpy as np


class FtLoss(object):
    def __init__(self, 
                 args: argparse.Namespace, 
                 subset: str):

        self.pre = args.pre
        self.subset = subset
        self.accumulation = args.accumulation
        
        pos_weight = self.get_pos_weight(args) if self.pre else None
        
        if self.pre:
            self.loss_fn = bce_calculation(pos_weight)
        else:
            self.loss_fn = ce_calculation()
            
    def get_pos_weight(self, args):
        cnt_path = os.path.join(args.input_path, f'interim/{args.task_name[:2]}/cnt.pkl')
        cnt_size = pickle_file_load(cnt_path)[args.task_name[-2:].lower()]
        mean_size = np.mean(cnt_size)
        pos_weights = mean_size / cnt_size
        pos_weights = np.maximum(1.0, np.minimum(10.0, pos_weights))
        pos_weights = torch.from_numpy(pos_weights.astype(np.float32))
        return pos_weights.cuda(args.rank)
            
    def get_loss(self, 
                 net_output: Tuple, 
                 label: torch.tensor) -> Tuple:
        
        if self.pre:
            pred = torch.sigmoid(net_output)
        else:
            pred = torch.argmax(net_output, -1)
            
        features = [pred, label]
        loss = self.loss_fn(net_output, label) / self.accumulation
        
        self.loss = loss
        
        return loss, features
    
    @property
    def return_log_dict_per_step(self):
        return {f'{self.subset} Loss': self.loss * self.accumulation}