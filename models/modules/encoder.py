import logging
import os
import sys
import dgl
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, GATConv, HGTConv
from typing import List
from models import register_model

logger = logging.getLogger(__name__)

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.dirname(os.path.abspath(os.path.dirname(dirname)))
dirname = os.path.join(dirname, 'utils')
sys.path.append(dirname)

from utils.graph_utils import *

 
# Base Model to be inherited:
class GearNetBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_dim = args.g_hidden_dim
        self.dropout = args.g_dropout
        self.g_num_layers = args.g_num_layers
        self.submodule = args.submodule
        self.d_radius = None
        self.downstream = (args.model == 'ft_model')
    
        # For eq(2) or eq(4)
        self.shared_kernel1 = nn.ModuleDict({
                str(etype) : nn.Linear(self.hidden_dim, self.hidden_dim) \
                    for etype in range(args.edge_types_cnt)
            })
        self.shared_kernel2, self.shared_e2l_linear, self.shared_IEConvMLP = None, None, None
        self.init_layer()
          
    def init_layer(self):
        self.layers = nn.ModuleList()     
        lastlayer = False
        for i in range(self.g_num_layers):
            if i == self.g_num_layers-1:
                self.dropout= 0.
                lastlayer = True
            module = self.submodule(
                            self.hidden_dim,
                            self.shared_kernel1,
                            self.shared_kernel2,
                            self.shared_e2l_linear,
                            self.shared_IEConvMLP,
                            lastlayer, 
                            self.dropout, 
                            self.d_radius)
            self.layers.append(module)
    
    @classmethod
    def build_model(cls, args):
        return cls(args)
    
    @property
    def get_layerwise_output(self):
        layer_concat = torch.hstack(self.layerwise)
        return layer_concat
    
    def forward(self, 
                graph: dgl.heterograph, 
                line_graph: dgl.line_graph) -> dgl.heterograph:        
        
        self.layerwise = []
        for layer in self.layers:
            graph, line_graph = layer(graph, 
                                      line_graph)
            if self.downstream:
                self.layerwise.append(graph.ndata['hv'])
        
        return graph
        
###################   
# Actual 4 Models # 
###################   

@register_model('GearNet')  
class GearNetModel(GearNetBase):
    def __init__(self, args):
        super().__init__(args)
        pass

    @classmethod
    def build_model(cls, args):
        return cls(args)
   
        
@register_model('GearNet-IEConv')       
class GearNetIEConvModel(GearNetBase):
    def __init__(self, args):
        super().__init__(args) 
        self.d_radius = args.d_radius
        self.shared_IEConvMLP = nn.Sequential(
            nn.Linear(14, self.hidden_dim), 
            nn.ReLU()
        )
        self.init_layer()
        
    @classmethod
    def build_model(cls, args):
        return cls(args)
        
    
@register_model('GearNet-Edge')        
class GearNetEdgeModel(GearNetModel):
    def __init__(self, args):
        super().__init__(args)   
        self.shared_kernel2 = nn.ModuleDict({
                str(etype) : nn.Linear(self.hidden_dim, self.hidden_dim) \
                    for etype in range(8) # 8 angles 
            })
        self.shared_e2l_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.init_layer()
        
    @classmethod
    def build_model(cls, args):
        return cls(args)
    
        
@register_model('GearNet-Edge-IEConv')       
class GearNetEdgeIEConvModel(GearNetEdgeModel):
    def __init__(self, args):
        super().__init__(args)    
        self.d_radius = args.d_radius
        self.shared_IEConvMLP = nn.Sequential(
            nn.Linear(14, self.hidden_dim), 
            nn.ReLU()
        )
        self.init_layer()
        
    @classmethod
    def build_model(cls, args):
        return cls(args)
    

#%%

###################
# Baseline Models #
###################   

@register_model('GCN')
class GCNEncoder(nn.Module):
    def __init__(self, args):
        super(GCNEncoder, self).__init__()
        hidden_dim = args.g_hidden_dim
        self.g_num_layers = args.g_num_layers
        
        self.layers = nn.ModuleList()
        for l in range(self.g_num_layers):
            self.layers.append(GraphConv(in_feats=hidden_dim, 
                                        out_feats=hidden_dim,
                                        activation=F.relu if l != self.g_num_layers-1 else None))
       
        self.dropout = nn.Dropout(p=args.g_dropout)

    @classmethod
    def build_model(cls, args):
        return cls(args)
    
    @property
    def get_graph_nodes_cnt(self):
        return self.node_cnts
    
    @property
    def get_layerwise_output(self):
        layer_concat = torch.hstack(self.layerwise)
        return layer_concat

    def forward(self, g: dgl.graph) -> dgl.graph:
        self.layerwise = []
        
        g, self.node_cnts = update_input_homo(g)
        feat = g.ndata.pop('hv')

        for i, layer in enumerate(self.layers):
            feat = layer(g, feat)
            if i < len(self.layers)-1:
                feat = self.dropout(feat)
            self.layerwise.append(feat)

        return g


@register_model('GAT')
class GATEncoder(nn.Module):
    def __init__(self, args):
        super(GATEncoder, self).__init__()
        self.g_num_layers = args.g_num_layers
        self.gat_layers = nn.ModuleList()
        self.g_heads = ([args.g_num_heads] * self.g_num_layers)

        # input projection (no residual)
        self.gat_layers.append(GATConv(
                            in_feats=args.g_hidden_dim, 
                            out_feats=args.g_hidden_dim, 
                            num_heads=self.g_heads[0],
                            feat_drop=args.g_dropout, 
                            attn_drop=args.g_dropout, 
                            residual=False, 
                            activation=F.elu
                            ))

        # hidden layers
        for l in range(1, self.g_num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                in_feats=args.g_hidden_dim * self.g_heads[l-1], 
                out_feats=args.g_hidden_dim, 
                num_heads=self.g_heads[l], 
                feat_drop=args.g_dropout if l != self.g_num_layers-1 else 0., 
                attn_drop=args.g_dropout if l != self.g_num_layers-1 else 0., 
                residual=True if l == self.g_num_layers-1 else False, 
                activation=F.elu if l != self.g_num_layers-1 else None, 
            ))

    @classmethod
    def build_model(cls, args):
        return cls(args)

    @property
    def get_graph_nodes_cnt(self):
        return self.node_cnts

    @property
    def get_layerwise_output(self):
        layer_concat = torch.hstack(self.layerwise)
        return layer_concat

    def forward(self, g: dgl.graph) -> dgl.graph:
        self.layerwise = []
        
        g, self.node_cnts = update_input_homo(g)
        feat = g.ndata.pop('hv') 
        
        for l in range(self.g_num_layers): 
            if l < self.g_num_layers-1:
                feat = self.gat_layers[l](g, feat).flatten(1) 
            else:
                feat = self.gat_layers[-1](g, feat).mean(1)
            self.layerwise.append(feat)
             
        return g



@register_model('HGT')
class HGTEncoder(nn.Module):
    def __init__(self, args):
        super(HGTEncoder, self).__init__()
        self.g_num_layers = args.g_num_layers
        self.hgt_layers = nn.ModuleList()
        
        for l in range(self.g_num_layers):
            self.hgt_layers.append(HGTConv(
                in_size=args.g_hidden_dim, 
                head_size=args.g_hidden_dim, 
                num_heads=args.g_num_heads,
                num_ntypes=1, 
                num_etypes=args.edge_types_cnt,
                dropout=args.g_dropout if l != self.g_num_layers-1 else 0.
            ))

        # Send to lower-dim
        self.out = nn.Linear(args.g_hidden_dim * args.g_num_heads, args.g_hidden_dim) 

    @classmethod
    def build_model(cls, args):
        return cls(args)

    @property
    def get_graph_nodes_cnt(self):
        return self.node_cnts

    @property
    def get_layerwise_output(self):
        layer_concat = torch.hstack(self.layerwise)
        return layer_concat

    def align_arguments(self, g: dgl.heterograph) -> Tuple:
        device = g.device
        ntypes = torch.tensor([0] * g.num_nodes(), device=device)
        etypes = []
        for i, etype in enumerate(g.etypes):
            etypes.extend([i]*g[etype].num_edges())
        etypes = torch.tensor(etypes, device=device)
        return ntypes, etypes

    def forward(self, g: dgl.heterograph) -> List:
        self.layerwise = []
        ntype, etype = self.align_arguments(g)
        
        g, self.node_cnts = update_input_hetero(g)
        feat = g.ndata.pop('hv') 
        
        for l in range(self.g_num_layers):     
            feat = self.out(self.hgt_layers[l](g, feat, ntype, etype))
            self.layerwise.append(feat)
             
        return g
