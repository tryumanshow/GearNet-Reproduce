import os
import sys
import dgl
import logging
import torch
import torch.nn as nn
from typing import List
from models import register_model, MODEL_REGISTRY
from utils.graph_utils import update_input

logger = logging.getLogger(__name__)

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.join(dirname, 'utils')
sys.path.append(dirname)


@register_model("ft_model")
class DownstreamModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.task_name = args.task_name

        # Fit the dimension for node & edge at the start time !
        self.node_up_dim = nn.Linear(args.node_start_dim, args.g_hidden_dim)
        self.edge_up_dim = nn.Linear(args.edge_start_dim, args.g_hidden_dim)

        self.model = self._encoder_model.build_model(args)
        
        self.classifier =  nn.Sequential(
                nn.Linear(args.ft_mlp_dim, args.ft_mlp_dim),
                nn.ReLU(), 
                nn.Linear(args.ft_mlp_dim, args.ft_mlp_dim),
                nn.ReLU(), 
                nn.Linear(args.ft_mlp_dim, args.g_out_dim)
            )
        
        self.baseline = args.baseline
        
    @property
    def _encoder_model(self):
        if self.args.ft_encoder_type is not None:
            return MODEL_REGISTRY[self.args.ft_encoder_type]
        else:
            return None

    @classmethod
    def build_model(cls, args):
        return cls(args)
    
    def graph_up_dim(self, graph: dgl.graph) -> dgl.graph:
        edge_up_dim = {}
        graph.ndata['hv'] = self.node_up_dim(graph.ndata['hv'])
        for canonical in graph.canonical_etypes:
            edge_up_dim[canonical] = self.edge_up_dim(graph.edata['he'][canonical])
        graph.edata['he'] = edge_up_dim
        del edge_up_dim
        return graph
    
    def get_graph_nodes_cnt(self, graph: dgl.graph) -> List:
        ubc_graph = dgl.unbatch(graph)
        num_nodes = [x.num_nodes() for x in ubc_graph]
        return num_nodes

    def forward(self, 
            graph: dgl.graph, 
            device: torch.device) -> torch.tensor:    
        
        graph = graph.to(device)
        graph = self.graph_up_dim(graph)
        
        if not self.baseline: # GearNet(-Variants)
            line_graph = update_input(graph)
            graph = self.model(graph, line_graph)
            node_cnt = self.get_graph_nodes_cnt(graph)
        else: # GCN, GAT, HGT
            graph = self.model(graph)
            node_cnt = self.model.get_graph_nodes_cnt
            
        net_output = self.model.get_layerwise_output
        net_output = self.classifier(net_output)
        net_output = torch.split(net_output, node_cnt)
        net_output = [x.mean(0).unsqueeze(0) for x in net_output]         
        net_output = torch.cat(net_output, 0)    
        
        return net_output