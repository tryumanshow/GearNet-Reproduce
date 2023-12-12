from typing import Tuple
import torch
import dgl
import numpy as np


def update_input(g_hetero: dgl.heterograph) -> dgl.heterograph:
    graph = dgl.to_homogeneous(g_hetero, g_hetero.ndata, g_hetero.edata) # edge index is maintained.
    
    hetero_dict = {}
    
    line_graph = init_line_graph(graph) # Edge order in homogeneous graph is maintained in the node order of line graph
    bins = discretize(line_graph) # len(bins): # of edges in line graph
    src, dst = line_graph.edges()
    
    for idx in torch.arange(-1, 8): # 8 bins & -1 will be ignored at modeling step.
        bins_idx = (bins == idx).nonzero(as_tuple=True)[0]
        bin_src = src[bins_idx]
        bin_dst = dst[bins_idx]
        hetero_dict[('edge', f'{idx}', 'edge')] = (bin_src, bin_dst)
    
    new_hetero_graph = dgl.heterograph(hetero_dict) # Re-initialize line graph as heterogeneous graph
    new_hetero_graph.ndata['hv'] = line_graph.ndata['he']
    
    assert line_graph.num_nodes() == new_hetero_graph.num_nodes(), 'Wrong'
    
    return new_hetero_graph


def update_input_homo(g_hetero: dgl.heterograph) -> dgl.graph:
    graph_ubc = dgl.unbatch(g_hetero)
    node_cnts = [x.num_nodes() for x in graph_ubc]
    graph_ubc = [dgl.to_homogeneous(x, ndata=x.ndata, edata=x.edata) for x in graph_ubc]
    graph = dgl.batch(graph_ubc)
    return graph, node_cnts


def update_input_hetero(g_hetero: dgl.heterograph) -> dgl.graph:
    node_cnts = [x.num_nodes() for x in dgl.unbatch(g_hetero)]
    graph = dgl.to_homogeneous(g_hetero, ndata=g_hetero.ndata, edata=g_hetero.edata)
    return graph, node_cnts

  
def init_line_graph(graph: dgl.heterograph) -> dgl.LineGraph:
    transform = dgl.LineGraph()
    new_graph = transform(graph)
    return new_graph
    

def discretize(graph: dgl.LineGraph) -> torch.tensor:
    line1, line2 = graph.edges()
    line1_coords = graph.ndata['coords_cat'][line1]
    line2_coords = graph.ndata['coords_cat'][line2]
    line_coords = torch.cat([line1_coords, line2_coords], dim=-1)
    angles = get_angle_bw_3points(*torch.split(line_coords, 3, -1))
    bins = calculate_bins(angles)
    return bins
   

def get_angle_bw_3points(x: torch.tensor, 
                         y1: torch.tensor, 
                         y2: torch.tensor, 
                         z: torch.tensor) -> torch.tensor:
    """
    Base code from: 
    https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    
    Changed:
    1. numpy -> torch
    2. vector op. -> matrix op.
    """
    assert torch.equal(y1, y2), 'Strange'
    
    yx = x - y1
    yz = z - y2
    
    cosine_angle = torch.mul(yx, yz).sum(1) / (torch.linalg.norm(yx, dim=-1) * torch.linalg.norm(yz, dim=-1))
    angle = torch.arccos(cosine_angle)
    degree = torch.rad2deg(angle).to(torch.double)
    
    # Loop case ( i -> j -> i )
    degree = torch.where(degree == 0., -100., degree)
    
    # The sequential edge with distance = 0 case ( cannot define angle )
    # Though it is impossible to define the 'angle', sequential edge with distance 0 should also be trained. 
    # So, instead of removing the edge, I will categorize this edge as discretized angle being 0.
    degree = torch.nan_to_num(degree, 0.)
    
    return degree


def get_dihedral_bw_4points(coordinate: torch.tensor, 
                            vec1: torch.tensor, 
                            vec2: torch.tensor, 
                            vec3: torch.tensor) -> torch.tensor:
    """
    Base code from:
    https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
    
    Changed:
    1. numpy -> torch
    2. vector op. -> matrix op.
    
    Background:
    The range of Dihedral angle: [-180, 180]
    """
    assert torch.equal(vec1[1], vec2[0]) and torch.equal(vec2[1], vec3[0]), 'Strange'
    point1, point2 = coordinate[vec1[0]], coordinate[vec1[1]]
    point3, point4 = coordinate[vec3[0]], coordinate[vec3[1]]

    basis0 = -1.0*(point2 - point1)
    basis1 = point3 - point2
    basis2 = point4 - point3

    basis1 /= torch.linalg.norm(basis1, dim =-1).unsqueeze(-1)

    v = basis0 - torch.mul(torch.mul(basis0, basis1).sum(1).unsqueeze(-1), basis1)
    w = basis2 - torch.mul(torch.mul(basis2, basis1).sum(1).unsqueeze(-1), basis1)

    x = torch.mul(v, w).sum(1)
    y = torch.mul(torch.cross(basis1, v), w).sum(1)
    
    angle = torch.arctan2(y, x)
    degree = torch.rad2deg(angle).to(torch.double)
    
     # To bypass nan dihedral angles to be bucketized into 8 buckets.
    degree = torch.nan_to_num(degree, 1e3) # 1e3 > 180 => Will be excluded at the target of optimization
    
    return degree


def calculate_bins(angles: torch.tensor, mode='angle') -> torch.tensor:
    assert mode in ['angle', 'dihedral'], 'Only two versions of discretizing are supported.'
    start = 0. if mode == 'angle' else -180. # To facilitate bucketizing
    
    bins = torch.linspace(start, 180 + 1e-5, 9, device=angles.device)                
    buckets = torch.bucketize(angles, bins, right=True)-1 # right=True: [ , )
    return buckets

