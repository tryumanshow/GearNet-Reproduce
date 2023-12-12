import os
import sys
from typing import Tuple
import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score
from joblib import Parallel, delayed

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.dirname(os.path.abspath(os.path.dirname(dirname)))
dirname = [os.path.join(dirname, x) for x in ['utils', 'criterions']]
sys.path.extend(dirname)

from utils.file_utils import *
from criterions import FtLoss

class FtBaseMetric:
    def __init__(self, 
                args, 
                epoch, 
                subset):

        self.loss_logger = FtLoss(args, subset)
        self.epoch = epoch
        self.subset = subset.capitalize()
        self.cum_step = 0
        self.total_loss = 0.
        self.accumulation = args.accumulation
        
        self.correct = 0
        self.total = 0
         
        self.dict_keys = ['Epoch', f'Avg {self.subset} Loss']

    def update(self, loss, pred, label):
        raise NotImplementedError
    
    def show(self):
        return NotImplementedError
    
    def get_correct(self, pred, target):
        total_len = target.size(0)
        correct = (pred == target).sum().detach().cpu().item()
        return correct, total_len
            
    def return_loss(self, net_output, label):
        self.cum_step += 1
        return self.loss_logger.get_loss(net_output, label)
            
    @property
    def return_total_loss(self):
        return [self.total_loss / (self.cum_step / self.accumulation)]
    
    
#%%
# Metric for EC, GO
class FtPreMetric(FtBaseMetric):
    def __init__(self, 
                 args,
                 epoch,
                 subset):
        super().__init__(args,
                         epoch,
                         subset)
        self.parallel_cpus = args.cores
        self.label_cnt = args.g_out_dim
        self.dict_keys.extend([f'{self.subset} Fmax', f'{self.subset} AUPRpair'])
        self.pred, self.label = [], []
        
    @property
    def show(self):
        return self.log_dict[f'{self.subset} Fmax']
        
    def compute_f1_score_at_threshold(self, 
                                      label: np.array,
                                      pred: np.array, 
                                      threshold: float) -> float:
        """
        Code reference from:
        https://github.com/aws-samples/lm-gvp
        """
        n_proteins = label.shape[0]
        y_pred_bin = pred >= threshold 
        precision = []
        recall = []
        for i in range(n_proteins):
            if y_pred_bin[i].sum() > 0:
                precision_i = precision_score(label[i], y_pred_bin[i])
                precision.append(precision_i)

            recall_i = recall_score(label[i], y_pred_bin[i])
            recall.append(recall_i)

        precision, recall = np.mean(precision), np.mean(recall)
        
        fscore = 2 * precision * recall / (precision + recall)
        
        return fscore
    
    def evaluate_multilabel(self, 
                            pred: np.array, 
                            label: np.array,
                            n_thresholds=100) -> Tuple:
        
        pred = np.array(pred).reshape(-1, self.label_cnt)
        label = np.array(label).reshape(-1, self.label_cnt)
        micro_aupr = average_precision_score(label, pred, average="micro")

        y_pred_sig = 1 / (1 + np.exp(-pred))

        thresholds = np.linspace(0.0, 1.0, n_thresholds, endpoint=False)
        f_scores = Parallel(n_jobs=self.parallel_cpus, verbose=10)(
            delayed(self.compute_f1_score_at_threshold)(label, 
                                                        y_pred_sig, 
                                                        thresholds[i])
            for i in range(n_thresholds)
        )

        return np.nanmax(f_scores), micro_aupr
    
    def return_log_dict_for_ddp(self, loss_list, metric_list):
        f_max, micro_aupr = self.evaluate_multilabel(*metric_list)
        log_group = [self.epoch, loss_list[0], f_max, micro_aupr]
        self.log_dict = {key: log_group[idx] for idx, key in enumerate(self.dict_keys)}
        return self.log_dict
    
    def update(self, loss, pred, label):
        self.total_loss += loss.item()
        self.pred += pred.flatten().cpu().numpy().tolist()
        self.label += label.flatten().cpu().numpy().tolist()
    
    @property
    def return_total_output(self):
        loss_list = [self.total_loss / (self.cum_step / self.accumulation)]
        metric_list = [self.pred, self.label]
        return loss_list, metric_list
        
        
#%%
# Metric for FC, RC
class FtPostMetric(FtBaseMetric):
    def __init__(self, 
                 args,
                 epoch,
                 subset):
        super().__init__(args,
                         epoch,
                         subset)
        self.additional_key = f'{self.subset} Accuracy'
        self.dict_keys.append(self.additional_key)
        
    @property
    def show(self):
        return self.log_dict[f'{self.subset} Accuracy']
        
    def update(self, loss, pred, label):
        self.total_loss += loss.item()
        correct, total = self.get_correct(pred, label)
        self.correct += correct
        self.total += total
        
    @property
    def calculate_accuracy(self): 
        accuracy = round(self.correct_ / self.total_ * 100, 2) # override
        return accuracy
        
    @property
    def return_total_output(self):
        loss_list = [self.total_loss / (self.cum_step / self.accumulation)]
        metric_list = [self.correct, self.total]
        return loss_list, metric_list
     
    def return_log_dict_for_ddp(self, loss_list, metric_list):
        self.correct_, self.total_ = list(map(sum, metric_list)) # concatenated
        metric_list = self.calculate_accuracy
        log_group = [self.epoch, loss_list[0], metric_list]
        self.log_dict = {key: log_group[idx] for idx, key in enumerate(self.dict_keys)}
        return self.log_dict