import argparse
import logging
import logging.config
import random
import os
import sys
from datetime import datetime
from pytz import timezone

os.environ['NUMEXPR_MAX_THREADS'] = '128'

logging.basicConfig( 
    format="%(asctime)s | %(levelname)s %(name)s %(message)s)))",
    datefmt="%Y-%m-%d %H:%M:%S",
    level = os.environ.get("LOGLEVEL", "INFO").upper(),
    stream = sys.stdout
)
logger = logging.getLogger(__name__)

sys.path.append('/home/swryu/anaconda3/lib/python3.8/site-packages')

import torch
import numpy as np
import torch.multiprocessing as mp
from utils.arg_utils import *


def get_parser():
    parser = argparse.ArgumentParser() 

    parser.add_argument('--project_name', type=str, default='GearNet-Reproduce')
    parser.add_argument('--entity', type=str, default='swryu')

    # DDP configs
    parser.add_argument('--device_ids', type=str, nargs='+', default=[0])
    parser.add_argument('--use_ddp', type=str2bool, default=True)
    parser.add_argument('--master_addr', type=str, default='localhost')
    parser.add_argument('--master_port', type=str, default='12335')
    parser.add_argument('--n_node', type=int, default=1, help='Only a single node environment.')

    # Checkpoint configs
    parser.add_argument('--input_path', type=str, default='/home/swryu/downstream', 
                        choices=['/home/swryu/uniprot', '/home/swryu/downstream'])
    parser.add_argument('--save_dir', type=str, default='/home/swryu/gearnet_checkpoint')
    parser.add_argument('--save_prefix', type=str, default='checkpoint')
    parser.add_argument('--load_from_pretrained', type=str2bool, default=False)

    # Training arguments
    parser.add_argument('--train_bsz', type=int, default=4)
    parser.add_argument('--eval_bsz', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--accumulation', type=int, default=1)

    # Model args
    parser.add_argument(
        '--model', type=str, default='ft_model',
        choices=['pt_model', 'ft_model'],
        help='name of the model to be trained'
    )
    ## Common model args    
    parser.add_argument('--d_seq', type=int, default=3, help='sequential distance threshold')
    parser.add_argument('--d_radius', type=float, default=10.0)
    parser.add_argument('--k_neighbors', type=int, default=10)
    parser.add_argument('--d_long', type=int, default=5, help='Long range interaction cutoff')
    parser.add_argument('--g_num_layers', type=int, default=6)
    parser.add_argument('--g_hidden_dim', type=int, default=512, help='Set 512 as default.')
    parser.add_argument('--g_dropout', type=float, default=0.1)
    
    # Pretraining args
    parser.add_argument('--pt_encoder_type', type=str, default='GearNet-Edge', 
                        choices=['GearNet', 
                                 'GearNet-IEConv',
                                 'GearNet-Edge', 
                                 'GearNet-Edge-IEConv'], 
                        help='Pretrained GearNet-Edge will be used for EC, GO, Reaction, \
                            Pretrained GearNet-Edge-IEConv will be used for FoldClassification.')
    parser.add_argument('--pt_task', type=int, default=0, 
                        choices=list(range(5)), 
                        help='0: Multiview Contrast, \
                            1: Residue Type Prediction, \
                            2: Distance Prediction, \
                            3: Angle Prediction, \
                            4: Dihedral Prediction')
    parser.add_argument('--pt_optimizer', type=str, default='Adam')
    parser.add_argument('--pt_tau', type=float, default=0.07)
    parser.add_argument('--pt_multiview_masking_ratio', type=float, default=0.15)
    parser.add_argument('--pt_others_sampling_cnt', type=int, default=512)
    parser.add_argument('--pt_readout', type=str, default='mean')
    
    
    # Downstream args
    parser.add_argument('--ft_encoder_type', type=str, default='GearNet-Edge-IEConv', 
                        choices=['GearNet', 
                                 'GearNet-IEConv',
                                 'GearNet-Edge', 
                                 'GearNet-Edge-IEConv', 
                                 'GCN', 
                                 'GAT',
                                 'HGT'])
    parser.add_argument('--g_num_heads', type=int, default=1, 
                        help='Needed at attention-based GNN models such as GAT, HGT')
    
    parser.add_argument('--ft_task', type=int, default=4, 
                        choices=list(range(6)), 
                        help='0: EC Prediction \
                            1: GO-MF Prediction \
                            2: GO-BP Prediction \
                            3: GO-CC Prediction \
                            4: Fold Classification \
                            5: Reaction Classification'
                            )    
    
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', 
                        choices=['ReduceLROnPlateau', 'StepLR'])

    return parser


def main():

    torch.set_printoptions(precision=6)

    args = get_parser().parse_args()
    args = arg_constraint(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_device if args.use_ddp else args.whole_devices

    set_struct(vars(args))
    mp.set_sharing_strategy('file_system')
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)   
    torch.backends.cudnn.deterministic = True

    if 'pt' in args.model:
        from trainers import PtTrainer as Trainer
    else:
        from trainers import FtTrainer as Trainer
            
    trainer = Trainer(args)

    if args.use_ddp:
        trainer.init_mp_trainer()
    else:
        trainer.train()


def set_struct(cfg: dict):
    root = os.path.abspath(
        os.path.dirname(__file__)
    )

    now = datetime.now()
    now = now.astimezone(timezone('Asia/Seoul'))

    output_dir = os.path.join(
        root,
        "outputs",
        now.strftime("%Y-%m-%d"),
        now.strftime("%H-%M-%S")
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    os.chdir(output_dir)

    job_logging_cfg = {
        'version': 1,
        'formatters': {
            'simple': {
                'format': '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler', 'formatter': 'simple', 'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler', 'formatter': 'simple', 'filename': 'train.log'
            }
        },
        'root': {
            'level': 'INFO', 'handlers': ['console', 'file']
            },
        'disable_existing_loggers': False
    }
    logging.config.dictConfig(job_logging_cfg)

    cfg_dir = ".config"
    os.mkdir(cfg_dir)
    os.makedirs(cfg['save_dir'], exist_ok=True)

    with open(os.path.join(cfg_dir, "config.yaml"), "w") as f:
        for k, v in cfg.items():
            print("{}: {}".format(k, v), file=f)


if __name__ == '__main__':
    main()
    