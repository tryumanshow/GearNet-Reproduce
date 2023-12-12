from argparse import Namespace
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Dict, List, Union


logger = logging.getLogger(__name__)



class MultiTestset:
    def __init__(self, args: Namespace):
        
        self.logic = True
        
        if args.task_name == 'FC':
            self.key = ['test_fold', 'test_superfamily', 'test_family']
        elif args.pre:
            self.key = ['test'+resolution for resolution in ['30', '40', '50', '70', '95']]
        else:
            self.logic = False
    
    def make_config(self, item):
        self.dic = {}
        for key in self.key:
            self.dic[key] = item
    
    @property
    def applicable(self):
        return self.logic

    def update(self, 
               cfg: Union[Dict, List], 
               obj: str) -> Dict:
        
        if obj == 'drop_last':
            del cfg['test']
            self.make_config(False)
            cfg.update(self.dic)
            
        elif obj == 'batch_size':
            eval_bsz = cfg.pop('test')
            self.make_config(eval_bsz)
            cfg.update(self.dic)
            
        else: # dataloaders
            loader_dict = {key: cfg[i] for i, key in enumerate(self.key)}
            cfg = loader_dict
    
        return cfg


class info_nce_calculation(nn.Module):
    """
    Code Reference from: https://theaisummer.com/simclr/
    EMA update version later - !
    """
    def __init__(self, args, subset):
        super().__init__()
        self.bsz = args.train_bsz if subset == 'train' else args.eval_bsz
        self.tau = args.pt_tau
        self.mask = (~torch.eye(self.bsz * 2, self.bsz * 2, dtype=bool)).float()

    def calc_similarity_batch(self, a, b):
        representations = torch.cat([a, b], dim=0)
        return F.cosine_similarity(representations.unsqueeze(1), 
                                   representations.unsqueeze(0), dim=2)

    def forward(self, proj1, proj2): 
        self.device = proj1.device

        batch_size = proj1.shape[0]
        z_i = F.normalize(proj1, p=2, dim=1)
        z_j = F.normalize(proj2, p=2, dim=1)

        similarity_matrix = self.calc_similarity_batch(z_i, z_j)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.tau)

        try:
            denominator = self.mask.to(self.device) * torch.exp(similarity_matrix / self.tau)
        except: # In case when drop_last does not work in a DDP setting
            denominator = self.mask.to(self.device)[:similarity_matrix.size(0), 
                                                    :similarity_matrix.size(0)] * torch.exp(similarity_matrix / self.tau)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * self.bsz)
        
        return loss


class bce_calculation(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.fn = nn.MultiLabelSoftMarginLoss(weight=pos_weight, 
                                              reduction='none')  # W/o the necessity to flatten
    
    def forward(self, yhat, y):
        loss = self.fn(yhat, y)
        loss = torch.sum(loss) / len(y)
        return loss


class ce_calculation(nn.Module):
    def __init__(self):
        super().__init__()
        self.ignore_index = -100 # Intentional
        self.fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index, 
                                    reduction='none')
        
    def forward(self, yhat, y):
        loss = self.fn(yhat, y)
        loss = torch.sum(loss) / (len(y) - torch.sum(y == self.ignore_index))
        return loss
    

class mse_calculation(nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = nn.MSELoss(reduction='none')
        
    def forward(self, yhat, y):
        loss = self.fn(yhat, y)
        loss = torch.sum(loss) / len(y)
        return loss


class get_optimizer:
    def __init__(self, args, model):
        if 'pt' in args.model:
            self.optim = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            if args.pre: # EC, GO
                self.optim = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                               weight_decay=args.weight_decay)
            else: # FC, RC
                self.optim = torch.optim.SGD(model.parameters(), lr=args.lr, 
                                             weight_decay=args.weight_decay)
        
    @property
    def init(self):
        return self.optim


class get_scheduler:
    def __init__(self, optimizer, args):
        optimizer = optimizer
        if args.scheduler == 'ReduceLROnPlateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(optimizer, 
                                            factor=args.scheduler_factor,
                                            patience=args.scheduler_patience)
        else:
            from torch.optim.lr_scheduler import StepLR
            self.scheduler = StepLR(optimizer, 
                                step_size=args.scheduler_stepsize,
                                gamma=args.scheduler_gamma)
            
    @property
    def init(self):
        return self.scheduler


def should_stop_early(patience, 
                      valid_criteria: float,
                      standard='higher') -> bool:
    
    assert valid_criteria is not None, 'Sth wrong'

    def is_better(a, b):
        return a > b if standard == 'higher' else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_criteria, prev_best):
        should_stop_early.best = valid_criteria
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= patience:
            logger.info(
                "Perform early stopping since valid performance hasn't improved for last {} runs".format(
                    patience
                )
            )
            return True
        else:
            return False


def caching_cuda_memory():
    free_mem, total_mem = torch.cuda.mem_get_info()
    if free_mem < total_mem * 0.05:
        torch.cuda.empty_cache()


@contextmanager
def rename_logger(logger, new_name):
    old_name = logger.name
    if new_name is not None:
        logger.name = new_name
    yield logger
    logger.name = old_name