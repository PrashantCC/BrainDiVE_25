"""
    All flags.
    Date created: 8/25/19
    Python Version: 3.6
"""

__author__ = "Guy Gaziv"
__email__ = "guy.gaziv@weizmann.ac.il"

import os, numpy as np
from absl import flags

# import self_super_reconst

import sys
sys.path.append("/home/prashant/BrainDiVE/self_super_reconst")   #I made this update from the above line


from pathlib import Path
FLAGS = flags.FLAGS

# PROJECT_ROOT = str(Path(__file__).parents[1])
PROJECT_ROOT = "/data6/shubham/PC/data/results_jointencoder_master_gaziv"

placeholder_str = '<PLACEHOLDER>'

flags.DEFINE_list('gpu', ['0'], 'GPUs')
flags.DEFINE_enum("im_res", '112', ['112'], 'Image pixel resolution')
# flags.DEFINE_enum("im_res", '256', ['112', '256'], 'Image pixel resolution')
flags.DEFINE_string("checkpoint_out", '/data6/shubham/PC/data/results_jointencoder_master_gaziv/checkpoints/sub1/{}.pth.tar'.format(placeholder_str), "Checkpoint path")
flags.DEFINE_integer('may_save', 0, '')
flags.DEFINE_integer("savebestonly", 1, 'Save checkpoint only if it is the best, otherwise do not save.')
flags.DEFINE_integer("sbj_num", 1, '')
flags.DEFINE_integer('separable', 0, 'Separable (Space-Feature) Encoder.')
flags.DEFINE_integer("is_rgbd", 0, '1: RGBD | 2: Depth only')
flags.DEFINE_integer("norm_within_img", 0, 'normalize within each depth map')

flags.DEFINE_string("select_voxels", '', 'File with voxels selected for analysis')

im_res = lambda : int(FLAGS.im_res)

def num_workers():
    return FLAGS.num_workers_gpu * len(FLAGS.gpu)

def get_checkpoint_out():
    if placeholder_str in FLAGS.checkpoint_out:
        return FLAGS.checkpoint_out.replace(placeholder_str, FLAGS.exp_prefix)
    else:
        return FLAGS.checkpoint_out

if __name__ == '__main__':
    pass
