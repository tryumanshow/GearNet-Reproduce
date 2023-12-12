import logging
import os
import sys
import torch
import dgl
import torch.nn as nn
import dgl.function as fn
from typing import Tuple
import warnings

# warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.dirname(os.path.abspath(os.path.dirname(dirname)))
dirname = os.path.join(dirname, 'utils')
sys.path.append(dirname)

from utils.graph_utils import *
   
class GearNetBaseLayer(nn.Module):
    def __init__(self,
                hidden_dim,
                shared_kernel1,
                shared_kernel2,
                shared_e2l_linear,
                shared_IEConvMLP, 
                last_layer,
                dropout, 
                d_radius):
        super().__init__()
   
        self.hidden_dim = hidden_dim
        
        # Shared kernel for Eq(2) or Eq(4) 
        self.kernel1 = shared_kernel1
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        
        # Shared kernel for Eq(3)
        self.kernel2 = shared_kernel2 
        self.bn2 = nn.BatchNorm1d(hidden_dim) if shared_kernel2 is not None else None
        self.e2n_linear = shared_e2l_linear if shared_kernel2 is not None else None # To be used on eq(4)
        
        self.shared_IEConvMLP = shared_IEConvMLP
        
        # Etc config
        self.last_layer = last_layer
        self.dropout = nn.Dropout(p=dropout)
        self.d_radius = d_radius
        
        
    #%% 
    #####################################
    # Vanilla Message Passing Related   #
    # Used at: GearNet(-IEConv) [Eq(2)] #
    #####################################
        
    def original_graph_update(self, nodes):
        updated = self.relu(self.bn1(nodes.data['eu_sum']))
        updated = self.dropout(updated)
        return {'hu': updated}

    def node_update(self, node):
        update = {'hv': node.data['hu'] + node.data['hv']} if self.last_layer \
                                                        else {'hv': node.data['hu']}
        return update
        
    def vanilla_forward(self,
                graph: dgl.heterograph) -> Tuple:
        
        with graph.local_scope():
            node_mp_func = {}
            for idx, etype in enumerate(graph.etypes): # 7 edges ( from 'knn' to 'sequential2' )
                graph.ndata[f'hv_{etype}'] = self.kernel1[str(idx)](graph.ndata['hv'])
                node_mp_func[etype] = (fn.copy_u(f'hv_{etype}', 'm'), 
                                    fn.sum('m', 'eu_sum'))
            graph.multi_update_all(node_mp_func, 'sum')
            graph.apply_nodes(self.original_graph_update) # hu
            graph.apply_nodes(self.node_update) # hu ( + hv ) -> hv
        
            return graph.ndata['hv']
        
        
    #%% 
    ###############################################
    # Edge Message Passing Related                #
    # Used at: GearNet-Edge(-IEConv) [Eq(3),(4)] #
    ###############################################

    def line_graph_update(self, nodes):
        updated = self.relu(self.bn2(nodes.data['r_msg']))
        updated = self.dropout(updated)
        return {'hv': updated}
    
    def linear_he(self, edges):
        return {'he': self.e2n_linear(edges.data['he_new'])} # Fully Connected in advance

    def override_he(self, graph):
        for etype in graph.etypes:
            graph.apply_edges(self.linear_he, etype=etype)
        del graph.edata['he_new']
        return graph

    def edge_mp_forward(self,
                graph: dgl.heterograph, 
                line_graph: dgl.line_graph) -> Tuple:
           
        with graph.local_scope() and line_graph.local_scope():    
            # Edge Message Passing ( Eq(3) )
            edge_mp_func = {}
            for etype in line_graph.etypes: # 8 edges ( 8 discretized angles )
                if etype == '-1': # i -> j -> i case: excluded on update
                    continue
                updated = self.kernel2[etype](line_graph.ndata['hv'])
                line_graph.ndata[etype] = updated
                edge_mp_func[etype] = (fn.copy_u(etype, 'm'), 
                                    fn.sum('m', 'r_msg'))
            line_graph.multi_update_all(edge_mp_func, 'sum')
            line_graph.apply_nodes(self.line_graph_update)
                
            # Update the edge features of original heterogeneous graph with updated edge features
            edge_meta_info = np.cumsum([graph[x].num_edges() for x in graph.etypes])
            edge_meta_info = np.insert(edge_meta_info, 0, 0)
            ug_v_split = torch.split(line_graph.ndata['hv'], \
                                                list(np.diff(edge_meta_info)))
            node_mp_msg = {}
            for idx, canonical in enumerate(graph.canonical_etypes):
                node_mp_msg[canonical] = ug_v_split[idx] 
            graph.edata['he_new'] = node_mp_msg
            
            # Node Message Passing ( Eq(4) )
            node_mp_func = {}
            graph = self.override_he(graph)
            for idx, etype in enumerate(graph.etypes): # 7 edges ( from 'knn' to 'sequential2' )
                updated = self.kernel1[str(idx)](graph[etype].edata['he'])
                graph[etype].edata['he_updated'] = updated
                graph.ndata[f'hv_{etype}'] = self.kernel1[str(idx)](graph.ndata['hv'])
                node_mp_func[etype] = (fn.e_add_u(f'he_updated', f'hv_{etype}', 'eu'), 
                                    fn.sum('eu', 'eu_sum'))
            graph.multi_update_all(node_mp_func, 'sum')
            graph.apply_nodes(self.original_graph_update) # hu
            graph.apply_nodes(self.node_update) # hu ( + hv ) -> hv
            
            return graph.ndata['hv'], line_graph.ndata['hv']
    
    
    #%% 
    ##########################################
    # IE-Conv Passing Related                #
    # Used at: GearNet(-Edge)-IEConv [Eq(6)] #
    ##########################################
        
    # t: gives 3 dim 
    def t_vector_in_radius(self, edges):
        return {'t': (edges.dst['coordinate'] - edges.src['coordinate']) / self.d_radius}
    
    def t_vector_not_in_radius(self, edges):
        return {'t': torch.zeros_like(edges.src['coordinate'])}
    
    # r: gives 3 dim
    def r_vector(self, edges):
        src, dst = edges.src['local_frame'], edges.dst['local_frame']
        return {'r': (src * dst).sum(-1)}
    
    # 𝛿:  gives 1 dim
    def delta_path(self, edges):
        src, dst, _ = edges.edges()
        d = (torch.abs(dst-src) / self.num_nodes).unsqueeze(-1)
        return {'d': d }
    
    def get_h_tilde(self, edges):
        cat_features = torch.cat((edges.data['t'], edges.data['r'], edges.data['d']), dim=-1)
        cat_features_prime = 1 - 2 * torch.abs(cat_features)
        cat_features = torch.cat((cat_features, cat_features_prime), dim=-1)
        cat_features = self.shared_IEConvMLP(cat_features)
        return {'ieconv_feat': cat_features}

    def ieconv_forward(self, 
                    graph: dgl.heterograph) -> dgl.heterograph:
        
        with graph.local_scope():
            self.num_nodes = graph.num_nodes()
            for etype in graph.etypes:
                if etype == 'radius':
                    graph.apply_edges(self.t_vector_in_radius, etype=etype)
                else:
                    graph.apply_edges(self.t_vector_not_in_radius, etype=etype)
            graph = dgl.to_homogeneous(graph, graph.ndata, graph.edata)
            graph.apply_edges(self.r_vector)
            graph.apply_edges(self.delta_path)
            graph.apply_edges(self.get_h_tilde)
            
            graph.update_all(fn.copy_e('ieconv_feat', 'm'), 
                            fn.sum('m', 'h_tilde'))
        
            return graph.ndata['h_tilde']
    
    def forward(self, 
                graph: dgl.heterograph, 
                line_graph: dgl.line_graph) -> Tuple:
        raise NotImplementedError
       
       
       
#%% 
class GearNetLayer(GearNetBaseLayer):
    def __init__(self, 
                hidden_dim,
                shared_kernel1,
                shared_kernel2,
                shared_e2l_linear,
                shared_IEConvMLP,
                last_layer,
                dropout, 
                d_radius):
        super().__init__(
            hidden_dim=hidden_dim,
            shared_kernel1=shared_kernel1,
            shared_kernel2=shared_kernel2,
            shared_e2l_linear=shared_e2l_linear,
            shared_IEConvMLP=shared_IEConvMLP,
            last_layer=last_layer,
            dropout=dropout,
            d_radius=d_radius
        )        
        pass
    
    
    def forward(self, 
            graph: dgl.heterograph, 
            line_graph: dgl.line_graph) -> Tuple:
        
        h_uv = self.vanilla_forward(graph) # hu ( + hv )
        graph.ndata['hv'] = h_uv
        del h_uv
        
        return graph, line_graph


       
#%%
       
class GearNetIEConvLayer(GearNetBaseLayer):
    def __init__(self, 
                hidden_dim,
                shared_kernel1,
                shared_kernel2,
                shared_e2l_linear,
                shared_IEConvMLP,
                last_layer,
                dropout, 
                d_radius):
        super().__init__(
            hidden_dim=hidden_dim,
            shared_kernel1=shared_kernel1,
            shared_kernel2=shared_kernel2,
            shared_e2l_linear=shared_e2l_linear,
            shared_IEConvMLP=shared_IEConvMLP,
            last_layer=last_layer,
            dropout=dropout,
            d_radius=d_radius
        )        
        pass

    def forward(self, 
            graph: dgl.heterograph, 
            line_graph: dgl.line_graph) -> Tuple:
    
        h_tilde = self.ieconv_forward(graph)  # h_tilde
        h_uv = self.vanilla_forward(graph)  # hu ( + hv )
        graph.ndata['hv'] = h_uv + h_tilde # hu ( + hv ) + h_tilde
        del h_tilde, h_uv
        
        return graph, line_graph


 
#%%   

class GearNetEdgeLayer(GearNetBaseLayer):
    def __init__(self, 
                hidden_dim,
                shared_kernel1,
                shared_kernel2,
                shared_e2l_linear,
                shared_IEConvMLP,
                last_layer,
                dropout, 
                d_radius):
        super().__init__(
            hidden_dim=hidden_dim,
            shared_kernel1=shared_kernel1,
            shared_kernel2=shared_kernel2,
            shared_e2l_linear=shared_e2l_linear,
            shared_IEConvMLP=shared_IEConvMLP,
            last_layer=last_layer,
            dropout=dropout,
            d_radius=d_radius
        )        
        pass

    def forward(self, 
            graph: dgl.heterograph, 
            line_graph: dgl.line_graph) -> Tuple:
    
        hv, hv_l = self.edge_mp_forward(graph, 
                                        line_graph)  # hu ( + hv ) , hv of line_graph
        graph.ndata['hv'], line_graph.ndata['hv'] = hv, hv_l
        del hv, hv_l
        
        return graph, line_graph
    
    
    
#%%  
  
class GearNetEdgeIEConvLayer(GearNetBaseLayer):
    def __init__(self, 
                hidden_dim,
                shared_kernel1,
                shared_kernel2,
                shared_e2l_linear,
                shared_IEConvMLP,
                last_layer,
                dropout, 
                d_radius):
        super().__init__(
            hidden_dim=hidden_dim,
            shared_kernel1=shared_kernel1,
            shared_kernel2=shared_kernel2,
            shared_e2l_linear=shared_e2l_linear,
            shared_IEConvMLP=shared_IEConvMLP,
            last_layer=last_layer,
            dropout=dropout,
            d_radius=d_radius
        )        
        pass
        

    def forward(self, 
                graph: dgl.heterograph, 
                line_graph: dgl.line_graph) -> Tuple:
        
        h_tilde = self.ieconv_forward(graph) # h_tilde
        hv, hv_l = self.edge_mp_forward(graph, 
                                    line_graph) # hu ( + hv ) , hv of line_graph
        graph.ndata['hv'], line_graph.ndata['hv'] = hv + h_tilde, hv_l # hu ( + hv ) + h_tilde, hv of line_graph
        del hv, h_tilde, hv_l
        
        return graph, line_graph
    
