
import sys
import os

sys.path.append("/data6/shubham/PC")
# os.environ["HF_HOME"] = "/ocean/projects/soc220007p/aluo/cache"  #I commented both the line

os.environ["HF_HOME"] = "/data6/shubham/PC/cache"

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
import encoder_model_vit_update
import nibabel as nib
import os
import time
import h5py
# from self_super_reconst import temp_load_enc
import paths


def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()


import random
from base64 import b64encode
random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))


def main(subject, functional1, region1, i1, j1, functional2, region2, i2, j2, gpu):

    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    print("Starting subject {}".format(str(subject)))

    print("Creating CLIP ViT")
    backbone = encoder_model_vit_update.feature_extractor_vit()  #I made this change from the above line
    backbone.eval()
    backbone.cuda()

    assert not backbone.training
    for name, param in backbone.named_parameters():
        param.requires_grad = False
    print("Created CLIP ViT and moved to GPU")
   
    try: del dataset   
    except: pass
        
    try: del brain_model    
    except: pass
        
    gc.collect()
    torch.cuda.empty_cache()

    functional_path1 = paths.mask_path.format(subject, functional1)
    functional_mask1 = load_from_nii(functional_path1)
    mask1 = (functional_mask1 == i1)
    n1 = np.sum(mask1)

    functional_path2 = paths.mask_path.format(subject, functional2)
    functional_mask2 = load_from_nii(functional_path2)
    mask2 = (functional_mask2 == i2)
    n2 = np.sum(mask2)

    size = {subject : n1 + n2}
    print(f"num of voxels = {size}")
    



    brain_model = encoder_model_vit_update.downproject_CLIP_split_linear_higher(size)
   
    weights_path = f"/data6/shubham/PC/data/results_jointencoder/checkpoints/subject_1_neurips_split_VIT_last_fully_linear/{functional1}_{functional2}/{region1}_{region2}/00100.chkpt"
    weights = torch.load(weights_path)
    brain_model.load_state_dict(weights["network"], strict=True)

    brain_model.cuda()
    brain_model.eval()
    for name, param in brain_model.named_parameters():
        param.requires_grad = False
    assert not brain_model.training

    try: del pipe    
    except: pass
        
    try: del pipe2    
    except: pass
        
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    repo_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16)
    # pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")  # I made this update as the the reccom. by the compiler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe2 = pipe.to("cuda")

    
    region_seeds = list(range(20000, 20010))

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


    def loss_function_brainDiVE(image_input):
        image_features = backbone(image_input)
        pred_response = brain_model.forward(image_features[0][0], image_features[0][1], image_features[1], [subject]).reshape(-1)
        pred_response1 = pred_response[:n1]
        pred_response2 = pred_response[n1:]
        ans = j1 * torch.mean(pred_response1) + j2 * torch.mean(pred_response2)
        return ans
    

    current_folder = f"/home/spshubham/prashant/joint_encoder_images/{functional1}_{functional2}/({str(j1)}){region1}_({str(j2)}){region2}"
    os.makedirs(current_folder, exist_ok=True)
    offset = 0

    pipe.brain_tweak = loss_function_brainDiVE

    for seed in region_seeds:
        print(f"starting {seed}")
        offset += 1
        print("Starting {}".format(str(offset).zfill(5)))
        if offset % 20 == 0:
            print("S{}, {} region, {}/500".format(subject, region1 + "_" + region2, offset))
            gc.collect()
        if offset % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
        image_name = os.path.join(current_folder, "{}_{}.png".format(region1 + "_" + region2, str(seed).zfill(12)))
        if os.path.exists(image_name):
            print("skipping")
            continue
        g = torch.Generator(device="cuda").manual_seed(int(seed))
        image = pipe("", sag_scale=0.75, guidance_scale=0.0, num_inference_steps=50, generator=g, clip_guidance_scale=130.0)
        if os.path.exists(image_name):
            continue
        image.images[0].save(image_name, format="PNG", compress_level=6)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "process subject")
    parser.add_argument('--subject_id', default="1", type = str)

    parser.add_argument('--functional1', default="floc-faces", type = str)
    parser.add_argument('--region1', default="FFA-1", type = str)
    parser.add_argument('--i1', default=2, type = int)
    parser.add_argument('--j1', default=1, type = float)

    parser.add_argument('--functional2', default="floc-places", type = str)
    parser.add_argument('--region2', default="PPA", type = str)
    parser.add_argument('--i2', default=2, type = int)
    parser.add_argument('--j2', default=0, type = float)

    parser.add_argument('--gpu', default="2", type = str)
    args = parser.parse_args()
    main(args.subject_id, args.functional1, args.region1, args.i1, args.j1, args.functional2, args.region2, args.i2, args.j2, args.gpu)