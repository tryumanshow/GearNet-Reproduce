from contextlib import nullcontext
import os
import sys
import wandb
import logging
import torch
import models
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel
from utils.trainer_utils import (
    MultiTestset,
    caching_cuda_memory,
    get_scheduler,
    get_optimizer,
    rename_logger,
    should_stop_early
)
from datasets import FtDataset
from utils.ddp_utils import *

logger = logging.getLogger(__name__)

dirname = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dirname = os.path.join(dirname, 'metrics')
sys.path.append(dirname)

from metrics import (
    FtPreMetric,
    FtPostMetric
)

class FtTrainer(object):
    def __init__(self, args):
        self.args = args

        self.input_path = args.input_path
        self.save_dir = args.save_dir
        self.save_prefix = args.save_prefix

        # DDP-related
        self.master_port = args.master_port

        # Training configs
        self.task = args.task_name
        self.thresholds = args.thresholds
        self.ie_conv = args.ie_conv
        self.lr = args.lr
        self.seed = args.seed
        self.n_epochs = args.n_epochs
        self.train_bsz = args.train_bsz
        self.valid_bsz = args.eval_bsz
        self.patience = args.patience
        self.accumulation = args.accumulation
        self.valid_subsets = ['valid'] 
        
        # Loading
        self.load_from_pretrained = args.load_from_pretrained

        self.start_epoch = 0
        self.pre = args.pre
        self.metric_logger = {
            True: FtPreMetric,
            False: FtPostMetric
        }
        self.MT = MultiTestset(args)
            
    def load_dataset(self, split: str):
        dataset = FtDataset(
            input_path=self.input_path,
            split=split, 
            thresholds=self.thresholds, 
            ie_conv=self.ie_conv, 
            task=self.task
        )
        return dataset

    def load_dataloader(self, dataset, sampler, split):
        batch_size = {
            'train': self.train_bsz,
            'valid': self.valid_bsz,
            'test': self.valid_bsz
        }
        if self.MT.applicable:
            batch_size = self.MT.update(batch_size, 'batch_size')

        data_loaders = DataLoader(
            dataset, 
            collate_fn=dataset.collator, 
            batch_size=batch_size[split], 
            num_workers=4, 
            pin_memory=True,
            sampler=sampler,
            worker_init_fn=seed_worker,
            generator=self.gen
        )

        return data_loaders

    def get_dataloader(self, split, rank, world_size):
        self.gen = torch.Generator()
        self.gen.manual_seed(self.seed)

        drop_last = {
            'train': True, 
            'valid': False, 
            'test': False
        }
        if self.MT.applicable:
            drop_last = self.MT.update(drop_last, 'drop_last')

        dataset = self.load_dataset(split)
        sampler = DistributedSampler(dataset, 
                                    rank=rank,
                                    num_replicas=world_size,
                                    seed=self.seed,
                                    drop_last=drop_last[split])
        loader = self.load_dataloader(dataset, sampler, split)

        return sampler, loader

    def define_dataloader(self, rank, world_size):
        self.train_sampler, self.train_loader = self.get_dataloader('train', rank, world_size)
        self.dev_loader = {'valid': self.get_dataloader('valid', rank, world_size)[-1]}
        if self.MT.applicable:
            loaders = [self.get_dataloader(x, rank, world_size)[-1] for x in self.MT.key]
            loaders = self.MT.update(loaders, 'dataloaders')
            self.dev_loader.update(loaders)
        else:
            self.test_loader = self.get_dataloader('test', rank, world_size)[-1]
            self.dev_loader.update({'test': self.test_loader})

    def init_mp_trainer(self):
        mp.spawn(self.train, nprocs=self.args.world_size, args=(self.args.world_size, ))

    def setup(self, rank, world_size, args):
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        logger.info(f"MASTERPORT: {args.master_port}") 
        dist.init_process_group('nccl', rank=rank, world_size=world_size)

    def set_device(self, rank, world_size):
        torch.cuda.set_device(rank)
        self.model.cuda(rank)
        self.model = DistributedDataParallel(self.model, 
                                            device_ids=[rank], 
                                            find_unused_parameters=True) 
        
        if self.load_from_pretrained:
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.cuda(rank)
        
        if rank == world_size-1:
            self.logger_print(world_size)

    def cleanup(self):
        dist.destroy_process_group()

    def logger_print(self, world_size):
        logger.info(
            "Num. model params: {:,} (Num. to be trained: {:,})".format(
                sum(p.numel() for p in self.model.parameters()),
                sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            )
        )

    def get_logger_list(self, data, world_size, obj):
        data = all_gather_list(data, 'downstream')
        data = all_gather_stats(data, world_size, obj)
        return data

    def load_pretrained(self):
        model_path = os.path.join(self.save_dir, f'{self.args.exp_name}_{self.save_prefix}_best.pt')
        self.checkpoint = torch.load(model_path, map_location='cpu') 
        
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        
        try:
            self.model.load_state_dict(self.checkpoint['model_state_dict'], strict=True)
        except:
            new_model_dict = deepcopy(self.checkpoint)
            new_model_state_dict = OrderedDict()
            for key, values in self.checkpoint['model_state_dict'].items():
                new_key = key.replace('module.', '')
                new_model_state_dict[new_key] = values
            new_model_dict['model_state_dict'] = new_model_state_dict
            self.checkpoint = new_model_dict
            self.model.load_state_dict(self.checkpoint['model_state_dict'], strict=True)
            self.start_epoch = self.checkpoint['epochs']

    def train(self, rank, world_size):

        self.args.rank = rank
        self.MASTER = (rank == world_size - 1) # Last gpu as a master

        if self.MASTER and 'pydevd' not in sys.modules:
            wandb.init(project=self.args.project_name, entity=self.args.entity)
            wandb.config.update(self.args)
            wandb.run.name = self.args.exp_name
            self.wandb = wandb
               
        self.model = models.build_model(self.args)
        self.optimizer = get_optimizer(self.args, self.model).init 
        self.scheduler = get_scheduler(self.optimizer, self.args).init

        if self.load_from_pretrained:
            self.load_pretrained()

        self.setup(rank, world_size, self.args)
        self.set_device(rank, world_size)
        self.define_dataloader(rank, world_size)

        for epoch in range(self.start_epoch+1, self.n_epochs+1):
            
            if not self.load_from_pretrained:
            
                logger.info(f"Begin training epoch {epoch} at rank {rank}")
                self.train_sampler.set_epoch(epoch)

                metric_logger = self.metric_logger[self.pre](self.args, epoch, 'train')

                self.model.train()
                print(f'Trainer loop start for rank {rank}')
                
                for idx, (graph, label) in enumerate(tqdm(self.train_loader, 
                                                        desc = f'Iteration time for train set', 
                                                        disable = not self.MASTER)):
                    
                    grad_sync = self.model.no_sync if (idx + 1) % self.accumulation != 0 else nullcontext
                    with grad_sync():
                    
                        net_output = self.model(graph, rank)
                        loss, features = metric_logger.return_loss(net_output, label.to(f'cuda:{rank}'))

                        if self.MASTER and 'pydevd' not in sys.modules:    
                            log_dict = metric_logger.loss_logger.return_log_dict_per_step
                            self.wandb.log(log_dict)

                        loss.backward()
                        if (idx+1) % self.accumulation == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                        
                    with torch.no_grad():
                        metric_logger.update(loss, *features)
                        
                    caching_cuda_memory()
        
                loss_list, metric_list = metric_logger.return_total_output
                
                dist.barrier()

                loss_list = self.get_logger_list(loss_list, world_size, 'loss')
                metric_list = self.get_logger_list(metric_list, world_size, 'metric')
                
                if self.MASTER:
                    self.train_log = metric_logger.return_log_dict_for_ddp(loss_list, metric_list)
                
                    with rename_logger(logger, "train"):
                        logger.info(
                            "Epoch: {}, Avg Train Loss: {:.3f}, Performance in Interest: {:.3f}".format(epoch, 
                                                                                                self.train_log['Avg Train Loss'],
                                                                                                metric_logger.show)
                        )
                    
                del metric_logger

            should_stop = self.validate_and_save(epoch, self.valid_subsets, rank, world_size) if not self.load_from_pretrained else None
  
            if (should_stop or (epoch == self.n_epochs)) or should_stop is None:
                if self.task == 'FC':
                    _ = self.validate_and_save(epoch, ['test_fold', 'test_superfamily', 'test_family'], rank, world_size)
                elif self.pre:
                    _ = self.validate_and_save(epoch, ['test30', 'test40', 'test50', 'test70', 'test95'], rank, world_size)
                else:
                    _ = self.validate_and_save(epoch, ['test'], rank, world_size)
                
                break

        self.cleanup()  
        logger.info('All the processes are done.')
        sys.exit('Deliberately ending system, because all the process is over.')

    def validate(
        self,
        epoch,
        valid_subsets, 
        rank, 
        world_size
    ):

        self.model.eval()
        for subset in valid_subsets:
            
            if 'test' in subset and (not self.load_from_pretrained):
                self.load_pretrained()
                self.model.to(f'cuda:{rank}')
            
            logger.info(f"Begin validation on {subset} subset at rank {rank}")

            metric_logger = self.metric_logger[self.pre](self.args, epoch, subset)
            dev_loader = self.dev_loader[subset]

            for _ , (graph, label) in enumerate(tqdm(dev_loader, 
                                                desc = f'Iteration time for {subset} set',
                                                disable = not self.MASTER)):
                with torch.no_grad():
                    net_output = self.model(graph, rank)
                    loss, features = metric_logger.return_loss(net_output, label.to(f'cuda:{rank}'))
                    metric_logger.update(loss, *features)
                
            loss_list, metric_list = metric_logger.return_total_output

            dist.barrier()

            loss_list = self.get_logger_list(loss_list, world_size, 'loss')
            metric_list = self.get_logger_list(metric_list, world_size, 'metric')
            
            eval_log = metric_logger.return_log_dict_for_ddp(loss_list, metric_list)
            subset_ = subset.capitalize()

            if self.MASTER:
                whole_log = {**self.train_log, **eval_log} if not self.load_from_pretrained else {**eval_log}
                
                if 'pydevd' not in sys.modules:
                    self.wandb.log(whole_log)

                with rename_logger(logger, subset):
                    logger.info(
                        "Epoch: {}, Avg {} Loss: {:.3f}, Performance in Interest: {:.3f}".format(
                            epoch, subset_, eval_log[f'Avg {subset_} Loss'], metric_logger.show
                        )
                    )

            del metric_logger

        return eval_log[f'Avg {subset_} Loss']

    def validate_and_save(
        self,
        epoch,
        valid_subsets, 
        rank,
        world_size
    ):
        os.makedirs(self.save_dir, exist_ok=True)

        should_stop = False
        valid_loss = self.validate(epoch, valid_subsets, rank, world_size)
        if 'valid' in valid_subsets:
            try: # ReduceLRonPlateau
                self.scheduler.step(valid_loss)
            except: # StepLR
                self.scheduler.step()
            
        should_stop |= should_stop_early(self.patience, valid_loss, 'lower')
        prev_best = getattr(should_stop_early, "best", None)

        if self.patience == 0:
            should_stop = False
        
        dist.barrier()
        
        if self.MASTER:
            
            if ((prev_best is None) or (prev_best == valid_loss)) and \
                'test' not in valid_subsets[0]:
                
                logger.info(
                    "Saving checkpoint to {}".format(
                        os.path.join(self.save_dir, f'{self.args.exp_name}_{self.save_prefix}_best.pt')
                    )
                )
                torch.save(
                    {
                        'model_state_dict': self.model.module.state_dict() if (
                            isinstance(self.model, DataParallel)
                        ) else self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'epochs': epoch, 
                        'args': self.args,
                    },
                    os.path.join(self.save_dir, f'{self.args.exp_name}_{self.save_prefix}_best.pt')
                )
                logger.info(
                    "Finished saving checkpoint to {}".format(
                        os.path.join(self.save_dir, f'{self.args.exp_name}_{self.save_prefix}_best.pt')
                    )
                )
            
            else:
                logger.info(
                    f"Loss is not proved at this epoch {epoch}. Prev: {prev_best}, Current: {valid_loss}"
                )

        return should_stop