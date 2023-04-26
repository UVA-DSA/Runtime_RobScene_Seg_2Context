from bdb import Breakpoint
import datetime
from os import path
import math
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from model.model import STCNModel
from dataset.static_dataset import StaticTransformDataset
from dataset.vos_dataset import VOSDataset

from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters




"""
Initial setup
"""
# Init distributed environment
#+distributed.init_process_group(backend="nccl")
# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

print('CUDA Device count: ', torch.cuda.device_count())

# Parse command line arguments
para = HyperParameters()
para.parse()

if para['benchmark']:
    torch.backends.cudnn.benchmark = True

local_rank = 0
"""
Model related
"""

if local_rank == 0:
    # Logging
    if para['id'].lower() != 'null':
        print('I will take the role of logging!')
        long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), para['id'])
    else:
        long_id = None
    logger = TensorboardLogger(para['id'], long_id)
    logger.log_string('hyperpara', str(para))
    #breakpoint()

    # Construct the rank 0 model
    model = STCNModel(para, logger=logger, save_path=path.join('saves', long_id, long_id) if long_id is not None else None, local_rank=local_rank).train()
else:
    # Construct model for other ranks
    model = STCNModel(para, local_rank=local_rank).train()

# Load pertrained model if needed
if para['load_model'] is not None:
    total_iter = model.load_model(para['load_model'])
    print('Previously trained model loaded!')
else:
    total_iter = 0

if para['load_network'] is not None:
    model.load_network(para['load_network'])
    print('Previously trained network loaded!')

"""
Dataloader related
"""
# To re-seed the randomness everytime we start a worker
def worker_init_fn(worker_id): 
    return np.random.seed(torch.initial_seed()%(2**31) + worker_id + local_rank*100)

def construct_loader(dataset):
    train_sampler = None #torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)  #para['batch_size'][need to chaange back]
    #para['batch_size']
    train_loader = DataLoader(dataset, para['batch_size'], sampler=train_sampler, num_workers=para['num_workers'], 
                            worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True) # the batch size need to be smaller than the dataset len
    return train_sampler, train_loader

def renew_vos_loader(max_skip):
    #breakpoint()
    print(yv_root)
    #yv_dataset_S = VOSDataset(path.join(yv_root, 'Suturing'), max_skip//15,'Suturing' ,is_bl=False)
   # yv_dataset_NP = VOSDataset(path.join(yv_root, 'Needle_Passing'), max_skip//15, 'Needle_Passing',is_bl=False)
    yv_dataset_NT = VOSDataset(path.join(yv_root, 'Knot_Tying'), max_skip//15, 'Knot_Tying', is_bl=False)

    train_dataset = ConcatDataset([yv_dataset_NT]) #+[yv_dataset_NT]+[yv_dataset_S +[yv_dataset_NT]+[yv_dataset_S]


    print('dataset size: ', len(train_dataset))
    print('Renewed with skip: ', max_skip)

    return construct_loader(train_dataset)



"""
Dataset related
"""

"""
These define the training schedule of the distance between frames
We will switch to skip_values[i] once we pass the percentage specified by increase_skip_fraction[i]
Not effective for stage 0 training
"""
skip_values = [15*2, 15*3, 15*4, 15*5, 15]#[10, 15, 20, 25, 5]

if para['stage'] == 0:
    static_root = path.expanduser(para['jigsaws'])
    #breakpoint()
    JIGSAWS_dataset = StaticTransformDataset(path.join(static_root), method=1)

    train_dataset = JIGSAWS_dataset
    train_sampler, train_loader = construct_loader(train_dataset)

    print('Static dataset size: ', len(train_dataset))
else:
    # stage 2 or 3
    increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.9, 1.0]
    yv_root = path.expanduser(para['jigsaws'])


    train_sampler, train_loader = renew_vos_loader(5)
    renew_loader = renew_vos_loader


"""
Determine current/max epoch
"""
## todo fill in the epochs
total_epoch = 10#200#500
#breakpoint()
current_epoch = 0
print('Number of training epochs (the last epoch might not complete): ', total_epoch)
if para['stage'] != 0:
    increase_skip_epoch = [round(total_epoch*f) for f in increase_skip_fraction]
    # Skip will only change after an epoch, not in the middle
    print('The skip value will increase approximately at the following epochs: ', increase_skip_epoch[:-1])

"""
Starts training
"""
# Need this to select random bases in different workers
np.random.seed(np.random.randint(2**30-1) + local_rank*100)
print('start training')
startime = time.time()
de =20
try:
    for e in range(current_epoch, total_epoch): #total_epoch
        print('Epoch %d/%d' % (e, total_epoch))
        if para['stage']!=0 and e!=total_epoch and e>=increase_skip_epoch[0]:
            while e >= increase_skip_epoch[0]:
                cur_skip = skip_values[0]
                skip_values = skip_values[1:]
                increase_skip_epoch = increase_skip_epoch[1:]
            print('Increasing skip to: ', cur_skip)
            train_sampler, train_loader = renew_loader(cur_skip)

        # Train loop
        model.train()
        
        for i,data in enumerate(train_loader):
            idx = 1
            if i==len(train_loader) -1:
                idx =e
            model.do_pass(data, idx) #total_iter
            #print(e)


        if e%de==0:
            endtime = time.time()
            mins = time.strftime("%H:%M:%S", time.gmtime(endtime-startime))
            startime = time.time()

            print(f'{de} epochs takes {mins} h:m:s')
    model.save(e)

finally:
    if not para['debug'] and model.logger is not None:
        model.save(total_iter)
    # Clean up
