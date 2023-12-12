from itertools import chain
import os
import sys

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.dirname(os.path.abspath(os.path.dirname(dirname)))
dirname = [os.path.join(dirname, x) for x in ['utils', 'criterions']]
sys.path.extend(dirname)

from utils.file_utils import *
from criterions import PtLoss

class PtMetric:
    def __init__(self, 
                args, 
                epoch, 
                subset):

        self.loss_logger = PtLoss(args, subset)
        self.epoch = epoch
        self.subset = subset.capitalize()
        self.cum_step = 0
        self.total_loss = 0.
        self.accumulation = args.accumulation
        
        self.dict_keys = ['Epoch', f'Avg {self.subset} Loss']

    def update(self, loss):
        self.total_loss += loss.item()
            
    def return_loss(self, net_output):
        self.cum_step += 1
        return self.loss_logger.get_loss(net_output)
            
    def return_log_dict(self):
        return {'Epoch': self.epoch, 
                f'Avg {self.subset} Loss': self.total_loss / self.cum_step}
    
    @property
    def return_total_loss(self):
        return [self.total_loss / (self.cum_step / self.accumulation)]
    
    def return_log_dict_for_ddp(self, loss_list):
        list_cat = [self.epoch] + loss_list
        log_dict = {key: list_cat[idx] for idx, key in enumerate(self.dict_keys)}
        if self.subset != 'Train':
            del log_dict['Epoch']
        return log_dict