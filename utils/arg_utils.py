import yaml
import argparse
import socket
from utils.file_utils import *
from models.modules.submodules import *


PT_LAYER = {
    'GearNet': GearNetLayer, 
    'GearNet-IEConv': GearNetIEConvLayer, 
    'GearNet-Edge': GearNetEdgeLayer, 
    'GearNet-Edge-IEConv': GearNetEdgeIEConvLayer, 
}

FT_LAYER = {}
FT_LAYER.update(PT_LAYER)


def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() == 'true': 
        return True 
    elif v.lower() == 'false':
         return False 
    elif v is None:
        return None
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')


def none_or_str(v):
    if v == 'None':
        return None
    return v
    

def exp_restriction(args):
    if 'pt' in args.model:
        assert 'uniprot' in args.input_path, 'Path Error'
    else:
        assert 'downstream' in args.input_path, 'Path Error'
    
    
def set_device(args):
    device_num = [str(x) for x in args.device_ids]
    args.world_size = len(args.device_ids) * args.n_node

    visible_device = ','.join(device_num)
    args.visible_device = visible_device

    if not args.use_ddp:
        devices = ''
        for i in range(torch.cuda.device_count()):
            devices += (str(i) + ', ')
        devices = devices[:-2]
        args.whole_devices = devices


def device_constraint(n):
    if (n & (n - 1)):
        return False
    else:
        return True


def model_constraint(args):
    
    with open('./utils/config.yaml') as f:
        config = yaml.safe_load(f)
    f.close()
    
    # Model-related
    assert args.g_num_layers > 1, 'The Number of Layers should be larger than 1.'
    
    # DDP-related
    if args.use_ddp:
        sock = socket.socket()
        sock.bind(('', 0))
        args.master_port = str(sock.getsockname()[1])
    else:
        args.rank = args.device_ids[0]
        assert len(args.device_ids) == 1, 'When you do not use DDP, it should be a single GPU.'

    # Dimension-related
    args.node_start_dim = 21
    args.edge_types_cnt = 2 * args.d_seq + 1
    args.edge_start_dim = 21 * 2 + args.edge_types_cnt + 1 + 1 
    
    # Training config-related
    args.thresholds = [args.d_seq, args.d_radius, args.k_neighbors, args.d_long]

    # Define task-related arguments
    if args.model == 'pt_model': # pre-train 
        CFG = config['PT']
        args.submodule = PT_LAYER[args.pt_encoder_type] 
        args.task_name = CFG['PT_NAME_MAPPING'][args.pt_task]
        args.loss_flag = CFG['LOSS_FLAG'][args.task_name]
        args.pt_mlp_dim = args.g_hidden_dim * CFG['PT_MLP_DIM'][args.task_name]
        args.pt_class = CFG['PT_TASK_CLASS'][args.task_name]
        
        args.lr = float(CFG['LR'])
        args.g_dropout = float(CFG['Dropout'])
        
        ####################
        # Hyper-parameters #
        ####################
        hyper = CFG['Hyperparam']
        HYPERPARAM = hyper[args.loss_flag]
        
        args.train_bsz = HYPERPARAM['train_bsz'][args.pt_encoder_type]
        if args.loss_flag == 'info_nce': # Multiview Contrast
            args.subseq_thres = HYPERPARAM['subseq_thres']
            args.subspace_thres = HYPERPARAM['subspace_thres']
            args.pt_multiview_masking_ratio = HYPERPARAM['pt_multiview_masking_ratio']
            args.pt_tau = HYPERPARAM['pt_tau']
        else: # Residue Type, Angle Prediction,  Dihedral Prediction, Distance Prediction
            args.pt_others_sampling_cnt = HYPERPARAM['sampling_cnt']
        
        args.ie_conv = True if 'IEConv' in args.pt_encoder_type else False

        if 'pydevd' in sys.modules:
            args.train_bsz = 4
            
        if args.use_ddp:
            args.train_bsz = args.train_bsz // len(args.device_ids)

        # To bypass OOM
        while args.train_bsz > 8:
            args.train_bsz = args.train_bsz // 2
            args.accumulation *= 2
            if args.train_bsz <= 8:
                break
    
    else:
        CFG = config['FT']
        args.task_name = CFG['FT_NAME_MAPPING'][args.ft_task]
        args.loss_flag = CFG['LOSS_FLAG'][args.task_name]
        args.ft_mlp_dim = args.g_hidden_dim * args.g_num_layers
        args.g_out_dim = CFG['FT_OUTPUT_SIZE'][args.task_name] 
        
        try:
            args.submodule = FT_LAYER[args.ft_encoder_type] 
        except: # GCN, GAT, HGT
            args.g_num_heads = CFG['BASELINE'][args.ft_encoder_type]['num_head']
        
        if any(task in args.task_name for task in ['EC', 'GO']):      
            TASK_CFG = CFG['EC_GO']  
            
            args.lr = float(TASK_CFG['LR'])
            args.g_dropout = TASK_CFG['Dropout']
            args.weight_decay = TASK_CFG['weight_decay']
            args.n_epochs = TASK_CFG['n_epochs']

            args.scheduler = TASK_CFG['scheduler']
            args.scheduler_factor = TASK_CFG['factor']
            args.scheduler_patience = TASK_CFG['patience']
            
            args.cores = TASK_CFG['cores']
            
            args.pre = True
                    
        else:
            TASK_CFG = CFG['FC_RC']
            
            args.lr = float(TASK_CFG['LR'])
            args.g_dropout = TASK_CFG['Dropout']
            args.weight_decay = float(TASK_CFG['weight_decay'])
            args.n_epochs = TASK_CFG['n_epochs']

            args.scheduler = TASK_CFG['scheduler']
            args.scheduler_stepsize = TASK_CFG['stepsize']
            args.scheduler_gamma = TASK_CFG['gamma']
            
            args.pre = False
        
        args.ie_conv = True if 'IEConv' in args.ft_encoder_type else False
        args.baseline = False if 'GearNet' in args.ft_encoder_type else True
        args.multi_testset = True if args.task_name in ['GO', 'FC']  else False
        
        # In case using 4 gpus are impossible.
        device_cnt = len(args.device_ids)
        assert device_constraint(device_cnt), 'The number of devices should be 2**n'
        args.train_bsz = 8 // device_cnt
        args.valid_bsz = args.train_bsz
        args.accumulation = device_cnt
    
    
def get_exp_name(args):
    encoder_type = args.pt_encoder_type if args.model == 'pt_model' else args.ft_encoder_type
    exp_name = [args.model, args.task_name, encoder_type] 
    args.exp_name = '_'.join(exp_name)
    
    
def arg_constraint(args):

    exp_restriction(args)
    set_device(args)
    model_constraint(args)  
    get_exp_name(args)

    return args


