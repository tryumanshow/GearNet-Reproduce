import os
import dgl
import logging
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances
from utils.file_utils import pickle_file_load 
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_path, 
        split, 
        thresholds, 
        ie_conv,
        task
    ):  
        """
        The order of the one-hot encoding:
        ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 
        'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 
        'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK']           
        
        - 'UNK' for unknown pretein <- Not actually exist in Swiss-Prot data
        """
        input_path_split = os.path.split(input_path)[0]
        stats_path = os.path.join(input_path_split, 'uniprot/interim', 'stats', f'molecule.pkl')
        _, self.onehot, self.mol_dict = pickle_file_load(stats_path)
        d_seq, self.d_radius, self.k_neighbors, self.d_long = thresholds
    
        self.d_seq_cand = list(range(-1*d_seq+1, d_seq))
        
        self.make_edge_name_for_one_hot
        self.ie_conv = ie_conv
        self.task=task

    @property
    def make_edge_name_for_one_hot(self):
        self.edge_name = [f'sequential{x}' for x in self.d_seq_cand]
        self.edge_name.extend(['radius', 'knn'])
        
        edge_name_ = np.array(self.edge_name).reshape(-1, 1)
        onehot = OneHotEncoder(sparse=False)
        onehot_ft = onehot.fit_transform(edge_name_)
        
        self.edge_idx_dict = {}
        for key, value in enumerate(self.edge_name):
            self.edge_idx_dict[value] = torch.tensor(onehot_ft[key], dtype=torch.float32).unsqueeze(0)
            
    def __len__(self):
        raise NotImplementedError()
    
    def __getitem__(self, index):
        raise NotImplementedError()

class PtDataset(BaseDataset):
    def __init__(
        self,
        input_path, 
        split, 
        thresholds, 
        ie_conv, 
        task
    ):
        super().__init__(
            input_path=input_path,
            split=split,
            thresholds=thresholds,
            ie_conv=ie_conv,
            task=task
        )
        
        self.data_path = os.path.join(input_path, 'interim', 'individual_data', split)
        print(f'Loading {task.upper()} Dataset for {split.capitalize()}...')

    def __len__(self):
        return len(os.listdir(self.data_path))
    
    def __getitem__(self, index):
        graph_path = os.path.join(self.data_path, f'index{index}.pkl')
        graph = pickle_file_load(graph_path)
        return {'graph': graph}

    def get_node_features(self, graph):
        amino_idx = [self.mol_dict[g[0]] for g in graph]
        amino_onehot = torch.tensor(self.onehot[amino_idx], dtype=torch.float32)
        return amino_onehot
    
    def _sequential_edge_idx_pick(self, graph_idx, threshold):
        graph_idx_ = [x+threshold for x in graph_idx]
        dst_info = [(i, x) for i, x in enumerate(graph_idx_) if x >= 0 and x < len(graph_idx)]
        dst_val, dst_idx = list(zip(*dst_info))
        src_idx = [graph_idx[i] for i in dst_val]
    
        src_idx = torch.LongTensor(src_idx)
        dst_idx = torch.LongTensor(dst_idx)
        
        assert src_idx.size() == dst_idx.size(), 'Wrong'
        
        return (src_idx, dst_idx)
    
    def _knn_edge_idx_stack(self, idx):
        idx_list = []
        for i, col_idcs in enumerate(idx):
            row_idcs = [i]*len(col_idcs)
            idx_list.append(list(zip(*[row_idcs, col_idcs])))
        idx_list = np.vstack(idx_list)
        return idx_list
    
    def get_sequential_edges(self, graph):
        sequential_edges = {}
        graph_idx = list(range(len(graph)))
        
        for candidate in self.d_seq_cand:
            idx = self._sequential_edge_idx_pick(graph_idx, candidate)
            sequential_edges[('atom', f'sequential{candidate}', 'atom')] = idx
        self.edge_dict.update(sequential_edges)
    
    def get_radius_knn_edges(self, coordinates, flag='radius'):
        self.ed = euclidean_distances(coordinates, coordinates) # already bidirectional
        
        def _common_filter(idx):
            # Filter out |i-j| < d_long ( self-loop is also removed by this criterion. )
            idx_diff = np.abs(np.diff(idx))
            idx = idx[(idx_diff >= self.d_long).flatten(), :]
            return idx, len(idx)==0
        
        if flag == 'radius':
            # Filter out whose euclidean distances is smaller than d_radius
            idx = np.argwhere(self.ed < self.d_radius)
            idx, empty = _common_filter(idx) # ex) 6TM5-Q in EC prediction of downstream task: Length 3 => all sequences are filtered out.
            src_idx, dst_idx = ([], []) if empty else (idx[:,0], idx[:,1])

        else:
            idx = np.argsort(self.ed, axis=1)
            idx = self._knn_edge_idx_stack(idx)
            idx, empty = _common_filter(idx)
            if empty: # ex) 6TM5-Q
                src_idx, dst_idx = [], []
            else:# Filter out k-neighbors
                row_idx = np.unique(idx[:, 0])
                idx = np.vstack([idx[idx[:, 0]==i][:self.k_neighbors] for i in row_idx])
                src_idx, dst_idx = idx[:,0], idx[:,1]
        
        src_idx = torch.LongTensor(src_idx)
        dst_idx = torch.LongTensor(dst_idx)
        
        dic2update = {('atom', f'{flag}', 'atom'): (src_idx, dst_idx)}
        self.edge_dict.update(dic2update)
            
    def initialize_edge_feature(self, g):
        
        def _exceptional(axis):
            return axis[:,None] if axis.dim() == 1 else axis[None,None]
        
        for en in self.edge_name:
            if g[en].num_edges() != 0:
                tmp_edges = g[en].edges()
                axis1 = g.ndata['hv'][tmp_edges[0]]
                axis2 = g.ndata['hv'][tmp_edges[1]]
                axis3 = self.edge_idx_dict[en].repeat(tmp_edges[0].size(0), 1)
                axis4 = torch.abs(tmp_edges[1] - tmp_edges[0]).unsqueeze(-1)
                axis5 = _exceptional(torch.tensor(self.ed[tmp_edges])) # Because of 6TM5-Q
                g.edges[en].data['he'] = torch.cat([axis1, axis2, axis3, axis4, axis5], axis=1)
                g.edges[en].data['coords_cat'] = torch.cat([g.ndata['coordinate'][tmp_edges[0]], 
                                                            g.ndata['coordinate'][tmp_edges[1]]], dim=-1) # [src, dst]
        return g

    def make_graph(self, graph):
        self.edge_dict = {}
        self.coordinates = {}
        
        coordinates = torch.tensor([g[1:] for g in graph], dtype=torch.float32)
        
        self.get_sequential_edges(graph)
        self.get_radius_knn_edges(coordinates)
        self.get_radius_knn_edges(coordinates, 'knn')
        
        g = dgl.heterograph(self.edge_dict)     
        g.ndata['coordinate'] = coordinates
        return g

    def get_orthonormal(self, g):
        """
        Concept was originally suggested at:
        https://www.mit.edu/~vgarg/GenerativeModelsForProteinDesign.pdf
        Again Used at:
        https://arxiv.org/pdf/2205.15675.pdf
        """
        
        def _pad_front_back(bn):
            zero_tensor = torch.zeros_like(bn)[0].unsqueeze(0)
            bn = torch.cat((zero_tensor, bn, zero_tensor), dim=0).unsqueeze(0)
            return bn
             
        coord = g.ndata['coordinate'] # x_i 
        coord_diff = coord[1:] - coord[:-1]
        u_coord = F.normalize(coord_diff, dim=-1) # u_i

        u_next = u_coord[1:] # u_{i+1}
        
        b = F.normalize(u_coord[:-1] - u_next, dim=-1) # b_i
        n = F.normalize(torch.cross(u_coord[:-1], u_next), dim = -1) # n_i
        bn = torch.cross(b, n) # b_i x n_i
        
        output = [_pad_front_back(x) for x in [b, n, bn]]
        output = torch.cat(output, dim=0).transpose(1, 0).contiguous()
        
        g.ndata['local_frame'] = output
        
        return g

    def wrap_graphs(self, graph):
        g = self.make_graph(graph)
        
        # Get node features
        node_features = self.get_node_features(graph)
        g.ndata['hv'] = node_features
        
        # Get edge features
        g = self.initialize_edge_feature(g)
        
        if self.ie_conv:
            g = self.get_orthonormal(g)

        return g

    def collator(self, samples):        
        graphs = [x['graph'] for x in samples]
        graphs = [self.wrap_graphs(x) for x in graphs] 
        graphs = dgl.batch(graphs)
        return graphs


#%% 
class FtDataset(PtDataset):
    def __init__(
        self,
        input_path, 
        split, 
        thresholds, 
        ie_conv, 
        task
    ):
        super().__init__(
            input_path=input_path,
            split=split,
            thresholds=thresholds,
            ie_conv=ie_conv, 
            task=task
        )

        task_ = 'GO' if 'GO' in task else task
        self.multilabel = any(task in self.task for task in ['EC', 'GO'])
 
        self.data_path = os.path.join(input_path, 'interim', task_, split)
        self.feature_path = os.path.join(self.data_path, 'feature')
        self.label_path = os.path.join(self.data_path, 'label')
        
    def __len__(self):
        return len(os.listdir(self.feature_path))
    
    def __getitem__(self, index):
        graph_path = os.path.join(self.feature_path, f'index{index}.pkl')
        label_path = os.path.join(self.label_path, f'index{index}.pkl')
        graph = pickle_file_load(graph_path)
        label = pickle_file_load(label_path)
   
        if self.multilabel:
            label = label[self.task.split('-')[-1].lower()]
            
        return {'graph': graph, 'label': label}

    def collator(self, samples):        
        graphs = [x['graph'] for x in samples]
        labels = [x['label'] for x in samples]
        
        graphs = [self.wrap_graphs(x) for x in graphs] 
        graphs = dgl.batch(graphs)
        
        if self.multilabel:  
            labels = np.vstack(labels)
        labels = torch.LongTensor(labels)
            
        return graphs, labels
