import os
import argparse
import dgl
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, OrderedDict, Tuple
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import euclidean_distances

from utils.file_utils import pickle_file_load
from utils.graph_utils import calculate_bins, get_angle_bw_3points, get_dihedral_bw_4points


#%%
class Noising:
    def __init__(self, args: argparse.Namespace):
        pass
    
    def augment(self, graph: dgl.graph, device: torch.device) -> Tuple:
        pass
        

#%%
class Augmentation(Noising):
    def __init__(self, args: argparse.Namespace):
        self.masking_ratio = args.pt_multiview_masking_ratio
        self.subseq_thres = args.subseq_thres
        self.subspace_thres = args.subspace_thres
        
    # Cropping Function #1: Subsequence
    def get_subsequence(self, graph: dgl.graph) -> dgl.graph:
        graph_cp = graph.clone()
        nodes2subgraph = list(range(self.center-self.subseq_thres+1, \
                                                self.center+self.subseq_thres))
        nodes2subgraph = torch.tensor([x for x in nodes2subgraph if x >= 0 and \
                                    x < graph.nodes()[-1].item()], device=self.device)
        graph_cp = graph_cp.subgraph(nodes2subgraph)
        return graph_cp
    
    # Cropping Function #2: Subspace
    def get_subspace(self, graph: dgl.graph) -> dgl.graph:
        graph_cp = graph.clone()
        coor = graph.ndata['coordinate'].cpu()
        distances = euclidean_distances(coor, coor)[self.center]
        nodes2subgraph = torch.arange(len(distances), device=self.device)[distances<=self.subspace_thres]
        graph_cp = graph_cp.subgraph(nodes2subgraph)
        return graph_cp
    
    # Transformation Function #1: Identity
    def get_identity(self, graph: dgl.graph) -> dgl.graph:
        graph_cp = graph.clone()
        return graph_cp
    
    # Transformation Function #2: Random Edge Masking
    def get_random_edge_masking(self, graph: dgl.graph) -> dgl.graph:
        graph_cp = graph.clone()
        transform = dgl.DropEdge(p=self.masking_ratio)
        graph_cp = transform(graph_cp)
        return graph_cp
    
    def random_center(self, graph: dgl.graph) -> int:
        center = random.randint(0, len(graph.nodes())-1)
        return center
    
    def cropping(self, 
                graph: dgl.graph, 
                number: int) -> dgl.graph:
        return {
            0: self.get_subsequence(graph),
            1: self.get_subspace(graph)   
        }[number]
        
    def transform(self, 
                graph: dgl.graph, 
                number: int) -> dgl.graph:
        return {
            0: self.get_identity(graph),
            1: self.get_random_edge_masking(graph)
        }[number]
        
    def augment_in_random(self, graph: dgl.graph) -> List[dgl.graph]:
        augmented_tmp = []
        augmented_list = []
        
        self.center = self.random_center(graph)
        
        croppings = [random.sample([0,1], 1)[0] for _ in range(2)]
        for cr in croppings:
            augmented = self.cropping(graph, cr)
            augmented_tmp.append(augmented)
        
        transforms = [random.sample([0,1], 1)[0] for _ in range(2)]
        for i, tf in enumerate(transforms):
            augmented = self.transform(augmented_tmp[i], tf)
            augmented_list.append(augmented)
            
        del augmented_tmp
        
        return augmented_list
    
    def augment(self, 
            graphs: dgl.graph, 
            device: torch.device) -> Tuple:
        
        self.device = device
        graphs = dgl.unbatch(graphs)
        augmented1, augmented2 = [], []
        
        for graph in graphs:
            aug1, aug2 = self.augment_in_random(graph)
            
            augmented1.append(aug1)
            augmented2.append(aug2)
    
        augmented1 = dgl.batch(augmented1)
        augmented2 = dgl.batch(augmented2)
        
        return augmented1, augmented2
    
    def post_process(self, 
                    augmented1: dgl.graph, 
                    augmented2: dgl.graph, 
                    mlp: nn.Module) -> Tuple:
        augmented1 = dgl.unbatch(augmented1)
        augmented2 = dgl.unbatch(augmented2)
        # Suppose Mean-Pooling
        augmented1 = [mlp(x.ndata['hv']).mean(0).unsqueeze(0) for x in augmented1] 
        augmented2 = [mlp(x.ndata['hv']).mean(0).unsqueeze(0) for x in augmented2]
        augmented1 = torch.vstack(augmented1)
        augmented2 = torch.vstack(augmented2)
        return augmented1, augmented2
    
    
#%%
class ResidueFeatureMasking(Noising):
    def __init__(self, args: argparse.Namespace):
        self.standard = args.pt_others_sampling_cnt
        stats_path = os.path.join(os.path.split(args.input_path)[0], 
                                  'uniprot/interim/stats', f'molecule.pkl')
        self.o_encoder, self.onehot, self.mol_dict = pickle_file_load(stats_path)
        
    def node_selection(self, graph: dgl.graph) -> List:
        try:
            all_nodes = np.random.choice(graph.num_nodes(), self.standard, replace=False)
        except: # Which can happen in small-size batch due to the OOM
            all_nodes = np.random.choice(graph.num_nodes(), self.standard//2, replace=False)
        all_nodes = torch.tensor(sorted(all_nodes), device=self.device)
        return all_nodes
        
    def augment(self, 
                graph: dgl.graph, 
                device: torch.device) -> Tuple:
        
        self.device = device
        graph_onehot = graph.ndata.pop('onehot')
        
        self.sampled_nodes_idx = self.node_selection(graph)
        
        graph.ndata['hv'][self.sampled_nodes_idx] = 0. # masking
        label = self.o_encoder.inverse_transform(graph_onehot[self.sampled_nodes_idx].cpu())
        label = list(label.flatten())
        label = torch.tensor([self.mol_dict[x] for x in label], device=device)

        return graph, label
    
    def post_process(self, 
                    augmented: dgl.graph,
                    mlp: nn.Module) -> torch.tensor:
        residue_feature = augmented.ndata.pop('hv')
        residue_feature = residue_feature[self.sampled_nodes_idx]
        residue_feature = mlp(residue_feature)
        return residue_feature
    
    
# %%
class EdgeDropping(Noising):
    def __init__(self, args: argparse.Namespace):
        self.standard = args.pt_others_sampling_cnt
        self.node_pair_list = {}
        self.label = []
        self.predicted = []
        
    @property
    def reinit(self):
        self.node_pair_list = {}
        self.label = []
        self.predicted = []
            
    # For Weighted Edge Sampling
    def get_per_edge_cnt(self) -> Dict:
        edges_identity = []
        for i, edge in enumerate(self.graph.etypes):
            idcs = [i] * self.graph.num_edges(edge)
            edges_identity.extend(idcs)
        all_edges = np.random.choice(edges_identity, self.standard, replace=False)
        all_edges = OrderedDict(sorted(Counter(all_edges).items()))
        return all_edges
            
    def get_distance(self, 
                    node_pairs: Tuple):
        coordinate = self.graph.ndata['coordinate']
        point1, point2 = coordinate[node_pairs[0]].cpu(), coordinate[node_pairs[1]].cpu()
        distance = euclidean_distances(point1, point2).diagonal()
        self.label.extend(distance)
        del coordinate, point1, point2
            
    def cut_edges_v1(self, 
                  edge_name: Tuple, 
                  edge_cnt: int):
        sampled_eids = np.random.choice(self.graph.num_edges(edge_name), edge_cnt, replace=False)
        sampled_eids = torch.tensor(sorted(sampled_eids), device=self.device)
        
        sampled_node_pairs = self.graph.find_edges(sampled_eids, etype=edge_name)
        self.node_pair_list.update({edge_name: sampled_node_pairs})
        
        self.get_distance(sampled_node_pairs)
        
        self.graph.remove_edges(sampled_eids, etype=edge_name)
            
    def augment(self, 
            graph: dgl.graph, 
            device: torch.device) -> Tuple:
        
        self.device = device
        self.graph = graph
        
        sample_num_per_edge = self.get_per_edge_cnt()
        for i, edge_name in enumerate(graph.canonical_etypes):
            self.cut_edges_v1(edge_name, sample_num_per_edge[i])
        
        self.label = torch.tensor(self.label, device=device)
        
        return self.graph, self.label
    
    def predict(self, graph, edge_name):
        edges = self.node_pair_list[edge_name]
        features = torch.cat((graph.ndata['hv'][edges[0]], graph.ndata['hv'][edges[1]]), dim=-1)
        self.predicted.append(features)
    
    def post_process(self, 
                    graph: dgl.graph,
                    mlp: nn.Module) -> torch.tensor:
        
        for edge_name in graph.canonical_etypes:
            self.predict(graph, edge_name)
            
        features = torch.vstack(self.predicted)
        features = mlp(features).flatten()
    
        self.reinit
        
        return features
    
    
#%%
class AdjacentEdgeDropping(EdgeDropping):
    def __init__(self, args):
        self.standard = args.pt_others_sampling_cnt
        self.neighbors2sample = 2
        self.neighbor_key = 0
        self.sampled_edges = defaultdict(OrderedDict)
        self.predicted = []
        self.node_triples = []
        
    @property
    def reinit(self):
        self.neighbor_key = 0
        self.sampled_edges = defaultdict(OrderedDict)
        self.predicted = []
        self.node_triples = []
                
    def random_next_candidates(self) -> Tuple:
        idx = np.random.choice(self.pop_idx, 1)[0]
        self.pop_idx.remove(idx)
        return idx, self.etypes[idx]

    def dict_update(self, 
                    etype: str, 
                    src: torch.tensor, 
                    dst: torch.tensor, 
                    ids: torch.tensor):
        exit_condition = etype in self.sampled_edges[self.neighbor_key].keys()
        while exit_condition:
            etype += "'" # To protect the key of dictionaries overlapped
            if etype not in self.sampled_edges[self.neighbor_key].keys():
                break
        self.sampled_edges[self.neighbor_key][etype] = (src, dst, ids)
    
    def sampled_src_dst_id(self, 
                           all_edges: Tuple, 
                           perm: torch.tensor, 
                           etype: str):
        src = all_edges[0][perm]
        dst = all_edges[1][perm]
        ids = all_edges[2][perm]
        self.dict_update(etype, src, dst, ids)
        
    def first_onehop_sampling(self):
        for i, etype in enumerate(self.etypes):
            all_edges = self.graph.all_edges(form='all', etype=etype)
            perm = torch.randperm(all_edges[-1].size(-1), device=self.device)
            perm = perm[:self.sample_num_per_edge[i]]
            self.sampled_src_dst_id(all_edges, perm, etype)
        self.neighbor_key += 1
             
    def onehop_edges_ratio(self):
        # Choose initial number of edges to sample based on the ratio of each edge types. 
        self.sample_num_per_edge = list(self.get_per_edge_cnt().values())
    
    def after_onehop_sampling(self):
        prev_info = self.sampled_edges[self.neighbor_key-1]
        for etype in prev_info.keys():
            self.pop_idx = list(range(len(self.etypes)))
            _, prev_dst, edges2exclude = prev_info[etype]
            # excluded_edges = {('atom', etype.replace("'", ''), 'atom'): edges2exclude}
            sg = dgl.sampling.sample_neighbors(g=self.graph, 
                                            nodes=prev_dst, 
                                            edge_dir='out',
                                            fanout=1)
                                            # exclude_edges=excluded_edges) # Protect from coming back to itself
            while True:
                _, random_next_path = self.random_next_candidates()
                next_edges = sg.all_edges(form='all', etype=random_next_path)
                try:
                    assert next_edges[0].size(0) == prev_dst.size(0), 'Out edge does not exist in some cases.'
                except:
                    continue # Select edges in majority
                break
            self.dict_update(random_next_path, *next_edges)
        self.neighbor_key += 1
    
    def sampling(self):
        self.onehop_edges_ratio()
        self.first_onehop_sampling()
        for _ in range(self.neighbors2sample-1):
            self.after_onehop_sampling()
            
    def extract_vector(self, 
                       vec1: Tuple, 
                       vec2: Tuple) -> torch.tensor:
        coordinate = self.graph.ndata['coordinate']
        x = coordinate[vec1[0]]; y1 = coordinate[vec1[1]]
        y2 = coordinate[vec2[0]]; z = coordinate[vec2[1]]
        return get_angle_bw_3points(x, y1, y2, z)
        
    def _augment(self, 
            graph: dgl.graph, 
            device: torch.device):
        self.graph = graph
        self.etypes = graph.etypes
        self.device = device
        self.sampling()

    def cut_edges_v2(self):
        # Double-Loop, but light
        update_place = self.sampled_edges[0]
        for i in range(1, len(self.sampled_edges)):
            edge_list = self.sampled_edges[i]
            for key, value in edge_list.items():
                key_ = key.replace("'", '')
                updated_src = torch.hstack((update_place[key_][0], value[0]))
                updated_dst = torch.hstack((update_place[key_][1], value[1]))
                updated_eid = torch.hstack((update_place[key_][2], value[2]))
                update_place[key_] = (updated_src, updated_dst, updated_eid)

        for key, value in update_place.items():
            self.graph.remove_edges(value[-1], etype=key)
        
    def augment(self, 
                graph: dgl.graph,
                device: torch.device) -> Tuple:
        
        self._augment(graph, device)     
        label = []
        
        for idx in range(len(self.sampled_edges[0])):
            vec1 = self.sampled_edges[0][list((self.sampled_edges[0]))[idx]]
            vec2 = self.sampled_edges[1][list((self.sampled_edges[1]))[idx]]
            angle = self.extract_vector(vec1, vec2)
            label.append(angle)
            node_idx = vec1[:2] + (vec2[1],)
            self.node_triples.append(node_idx)
            
        label = torch.hstack(label)
        label = calculate_bins(label)
        label[label < 0] = -100  # Do not optimize when angle is NaN
        
        self.cut_edges_v2()
            
        return self.graph, label
    
    def predict(self, 
                prev: torch.tensor, 
                middle: torch.tensor, 
                post: torch.tensor):
        features = torch.cat((self.nf[prev], self.nf[middle], self.nf[post]), -1)
        self.predicted.append(features)
        
    def post_process(self, 
                    graph: dgl.graph,
                    mlp: nn.Module) -> torch.tensor:
        
        self.nf = graph.ndata['hv'] # node feature
        
        for nt in self.node_triples:
            self.predict(*nt)
        
        features = torch.vstack(self.predicted)
        features = mlp(features)
    
        self.reinit
        
        return features
    
   
#%%
class TripleEdgeDropping(AdjacentEdgeDropping):
    def __init__(self, args):
        self.standard = args.pt_others_sampling_cnt
        self.neighbors2sample = 3
        self.neighbor_key = 0
        self.sampled_edges = defaultdict(OrderedDict)
        self.predicted = []
        self.node_quadruples = []
        
    @property
    def reinit(self):
        self.neighbor_key = 0
        self.sampled_edges = defaultdict(OrderedDict)
        self.predicted = []
        self.node_quadruples = []
        
    def extract_vector(self, 
                       vec1: Tuple, 
                       vec2: Tuple, 
                       vec3: Tuple) -> torch.tensor:
        coordinate = self.graph.ndata['coordinate']
        return get_dihedral_bw_4points(coordinate, vec1, vec2, vec3)
        
    def augment(self, 
                graph: dgl.graph,
                device: torch.device) -> Tuple:
        
        self._augment(graph, device)     
        label = []
        
        for idx in range(len(self.sampled_edges[0])):
            vec1 = self.sampled_edges[0][list((self.sampled_edges[0]))[idx]]
            vec2 = self.sampled_edges[1][list((self.sampled_edges[1]))[idx]]
            vec3 = self.sampled_edges[2][list((self.sampled_edges[2]))[idx]]
            dihedral = self.extract_vector(vec1, vec2, vec3)
            label.append(dihedral)
            node_idx = vec1[:2] + vec3[:2]
            self.node_quadruples.append(node_idx)

        label = torch.hstack(label)
        label = calculate_bins(label, 'dihedral')
        label[label==8] = -100 # Do not optimize angles when dihedral angle is NaN
        
        self.cut_edges_v2()
        
        return self.graph, label
    
    def predict(self, 
                prev: torch.tensor, 
                middle1: torch.tensor, 
                middle2: torch.tensor, 
                post: torch.tensor):
        features = torch.cat((self.nf[prev], self.nf[middle1], 
                              self.nf[middle2], self.nf[post]), -1)
        self.predicted.append(features)
    
    def post_process(self, 
                    graph: dgl.graph,
                    mlp: nn.Module) -> torch.tensor:
        
        self.nf = graph.ndata['hv'] # node feature
        
        for nt in self.node_quadruples:
            self.predict(*nt)
        
        features = torch.vstack(self.predicted)
        features = mlp(features)
    
        self.reinit
        
        return features