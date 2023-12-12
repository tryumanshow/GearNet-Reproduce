import argparse
from typing import Tuple
from utils.trainer_utils import (
    info_nce_calculation, 
    ce_calculation, 
    mse_calculation
)


class PtLoss(object):
    def __init__(self, 
                 args: argparse.Namespace, 
                 subset: str):

        self.subset = subset
        self.accumulation = args.accumulation
        
        if args.task_name == 'Multiview_Contrast':
            self.loss_fn = info_nce_calculation(args, subset)
            
        elif args.task_name in ['Residue_Type_Pred', 
                                'Angle_Pred', 
                                'Dihedral_Pred']:
            self.loss_fn = ce_calculation()
            
        else:
            self.loss_fn = mse_calculation()

    def get_loss(self, net_output: Tuple):
        self.loss = self.loss_fn(*net_output) 
        return self.loss 
    
    @property
    def return_log_dict_per_step(self):
        return {f'{self.subset} Loss': self.loss * self.accumulation}