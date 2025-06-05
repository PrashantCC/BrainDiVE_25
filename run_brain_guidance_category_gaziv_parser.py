import sys
import os

sys.path.append("/data6/home2/spshubham/prashant")

os.environ["HF_HOME"] = "/data6/home2/spshubham/prashant/cache"

# print(random.getstate())
# exit()
import timm
import argparse
import torch
from torchvision import models, transforms
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from encoder_dataloader_update import neural_loader
from brain_guide_pipeline import mypipelineSAG
import pickle
import gc
import encoder_model_vit
import nibabel as nib
import os
import time
import h5py
import paths
from matplotlib import pyplot as plt

from self_super_reconst import temp_load_enc

def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()



import random
from base64 import b64encode
random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))

# all_subjects = [1, 2, 5, 7]

# for subject in all_subjects:
def main(subject, region, gpu):

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    print("Starting subject {}".format(str(subject)))
   
    gc.collect()
    torch.cuda.empty_cache()

    try:
        del pipe
    except:
        pass

    try:
        del pipe2
    except:
        pass

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    repo_id = "stabilityai/stable-diffusion-2-1-base"

    pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")  # I made this update as the the reccom. by the compiler

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe2 = pipe.to("cuda")


    print(f"starting {region}")

    print("Starting S{} {}".format(subject, region))
    

    region_seeds = range(3000, 3010)


    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


    enc, num_voxels = temp_load_enc.load_encoder(subject, region)

    enc.cuda()
    enc.eval()

    for name, param in enc.named_parameters():
        param.requires_grad = False
    assert not enc.training

    def loss_function_gaziv(image_input):

        image = image_input.squeeze(0)
 
        all_act1 = temp_load_enc.get_activations(enc, image, num_voxels)

        ans = torch.mean(all_act1)
            
        return +ans
    
    current_folder = f"/data6/home2/spshubham/prashant/data/results_gaziv_algonauts/joint_encoder_master_images/sub{subject}/single_region/{region}"

    os.makedirs(current_folder, exist_ok=True)
    offset = 0

    pipe.brain_tweak = loss_function_gaziv
    
    for seed in region_seeds:
        print(f"starting {seed}")
        offset += 1
        print("Starting {}".format(str(offset).zfill(5)))
        if offset % 20 == 0:
            print("S{}, {} region, {}/500".format(subject, region, offset))
            gc.collect()
        if offset % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
        image_name = os.path.join(current_folder, "{}_{}.png".format(region, str(seed).zfill(12)))
        if os.path.exists(image_name):
            print("skipping")
            continue
        g = torch.Generator(device="cuda").manual_seed(int(seed))
        image, activity_vector = pipe("", sag_scale=0.75, guidance_scale=0.0, num_inference_steps=50, generator=g, clip_guidance_scale=130.0)
        if os.path.exists(image_name):
            continue
        image.images[0].save(image_name, format="PNG", compress_level=6)

        # plt.figure(figsize=(10, 6))
        # plt.plot(np.arange(50), activity_vector, color='red')
        # plt.xlabel('time steps')
        # plt.ylabel('activity')
        # plt.legend()
        # plt.grid(True)
        # plot_name = os.path.join(current_folder, "{}_{}_plot.png".format(region, str(seed).zfill(12)))
        # plt.savefig(plot_name)  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "process subject")
    parser.add_argument('--subject_id', default=1, type = int)

    parser.add_argument('--region', default="FFA-1", type = str)
    parser.add_argument('--gpu', default="6", type = str)
    args = parser.parse_args()
    main(args.subject_id, args.region, args.gpu)
