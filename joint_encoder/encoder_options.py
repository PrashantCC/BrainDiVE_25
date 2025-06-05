import argparse
import torch
import os
import random
import numpy as np
import paths

def bool_flag(s):
    if s == '1':
        return True
    elif s == '0':
        return False
    msg = 'Invalid value "%s" for bool flag (should be 0 or 1)'
    raise ValueError(msg % s)

def list_float_flag(s):
    return [float(_) for _ in list(s)]

class Options():

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        parser = self.parser
        parser.add_argument('--exp_name', default="subject_{}_neurips_split_v3") # We override this in the trainer anyways

        # parser.add_argument('--save_loc', default="/media/internal_8T/prashant/BrainDiVE/results2/checkpoints", type=str) # Where are weights saved
        parser.add_argument('--save_loc', default="/data6/shubham/PC/data/results_jointencoder/checkpoints", type=str) # Where are weights saved
        
        parser.add_argument('--subject_id', default=["1"], nargs='+') # Put just a single subject here
        parser.add_argument('--gpus', default=1, type=int) # Number of GPUs to use. Only tested on one
        parser.add_argument('--gpu_id', default=3, type=int) # Number of GPUs to use. Only tested on one

        # parser.add_argument('--neural_activity_path', default="/ocean/projects/soc220007p/aluo/data/cortex_subj_{}.npy")
        # parser.add_argument('--neural_activity_path', default="/media/internal_8T/prashant/BrainDiVE/NSD_zscored/cortex/cortex_subj_{}.npy")
        parser.add_argument('--neural_activity_path', default=paths.neural_path)
        
        # parser.add_argument('--image_path', default="/ocean/projects/soc220007p/aluo/data/image_data.h5py") # All images for all subjects in one h5py
        parser.add_argument('--image_path', default=paths.image_path) # All images for all subjects in one h5py



       

        parser.add_argument('--epochs', default=100, type=int) # Total epochs to train for, we use 100
        parser.add_argument('--resume', default=0, type=bool_flag) # Load weights or not from latest checkpoint
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--lr_init', default=3e-4, type=float)  # Starting learning rate for adam/adamw
        parser.add_argument('--lr_decay', default=5e-1, type=float)  # Learning rate decay rate, so at the end of training how much you want the last lr to be.

        parser.add_argument("--functional1", default = "floc-faces", type = str)
        parser.add_argument("--region1", default="FFA-1", type = str)
        parser.add_argument("--i1", default=2, type = int)

        parser.add_argument("--functional2", default = "floc-places", type = str)
        parser.add_argument("--region2", default="PPA", type = str)
        parser.add_argument("--i2", default=2, type = int)

    def parse(self):
        # initialize parser
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        torch.manual_seed(0)
        # random.seed(0)
        np.random.seed(0)
        args = vars(self.opt)
        print('| options')
        for k, v in args.items():
            print('%s: %s' % (str(k), str(v)))
        print()
        return self.opt