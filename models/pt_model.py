import dgl
import os
import sys
import logging
import torch
import torch.nn as nn
from typing import Tuple
from models import register_model, MODEL_REGISTRY
from utils.graph_utils import update_input

logger = logging.getLogger(__name__)

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.join(dirname, 'utils')
sys.path.append(dirname)

from utils.noising_utils import (
    Augmentation, 
    ResidueFeatureMasking, 
    EdgeDropping, 
    AdjacentEdgeDropping,
    TripleEdgeDropping
)

@register_model("pt_model")
class PretrainModel(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.task_name = args.task_name

        # Fit the dimension for node & edge at the start time !
        self.node_up_dim = nn.Linear(args.node_start_dim, args.g_hidden_dim)
        self.edge_up_dim = nn.Linear(args.edge_start_dim, args.g_hidden_dim)

        self.model = self._encoder_model.build_model(args)
        
        if args.task_name == 'Multiview_Contrast':
            self.noising = Augmentation(args)
            self.mlp = nn.Sequential(
                nn.Linear(args.pt_mlp_dim, args.pt_mlp_dim),
                nn.ReLU(), 
                nn.Linear(args.pt_mlp_dim, args.pt_mlp_dim)
            )
        else:
            if args.task_name == 'Residue_Type_Pred':
                self.noising = ResidueFeatureMasking(args)
            elif args.task_name == 'Distance_Pred':
                self.noising = EdgeDropping(args)
            elif args.task_name == 'Angle_Pred':
                self.noising = AdjacentEdgeDropping(args)
            else:
                self.noising = TripleEdgeDropping(args)
                
            self.mlp = nn.Linear(args.pt_mlp_dim, args.pt_class)
            
    @property
    def _encoder_model(self):
        if self.args.pt_encoder_type is not None:
            return MODEL_REGISTRY[self.args.pt_encoder_type]
        else:
            return None

    @classmethod
    def build_model(cls, args):
        return cls(args)
    
    def graph_up_dim(self, graph: dgl.graph) -> dgl.graph:
        if self.task_name == 'Residue_Type_Pred':
            graph.ndata['onehot'] = graph.ndata['hv']
        edge_up_dim = {}
        graph.ndata['hv'] = self.node_up_dim(graph.ndata['hv'])
        for canonical in graph.canonical_etypes:
            edge_up_dim[canonical] = self.edge_up_dim(graph.edata['he'][canonical])
        graph.edata['he'] = edge_up_dim
        del edge_up_dim
        return graph

    def forward(self, 
            graph: dgl.graph, 
            device: torch.device) -> Tuple:    
        
        graph = graph.to(device)
        graph = self.graph_up_dim(graph)
        
        if self.task_name == 'Multiview_Contrast':
            augmented1, augmented2 = self.noising.augment(graph, device)
            
            line_graph1 = update_input(augmented1)
            line_graph2 = update_input(augmented2)
            
            augmented1 = self.model(augmented1, line_graph1)
            augmented2 = self.model(augmented2, line_graph2)
            
            augmented1, augmented2 = self.noising.post_process(augmented1,
                                                            augmented2,
                                                            self.mlp)
            
            net_output = (augmented1, augmented2)
            
        else:
            augmented, label = self.noising.augment(graph, device)
            line_graph = update_input(augmented)
            augmented = self.model(augmented, line_graph)
            augmented = self.noising.post_process(augmented, self.mlp)
            
            net_output = (augmented, label)
            
        return net_output