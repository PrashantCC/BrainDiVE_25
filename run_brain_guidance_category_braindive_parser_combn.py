
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

# all_subjects = [1, 2, 5, 7]

# for subject in all_subjects:
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
        
    try: del brain_model1    
    except: pass

    try: del brain_model2    
    except: pass
        
    gc.collect()
    torch.cuda.empty_cache()

    # class myarg():
    #     def __init__(self):
    #         self.subject_id = subject
    #         self.functional = functional
    #         self.region = region
    #         self.i = i

    # other_args = myarg()
    # dataset = neural_loader(other_args)

    # functional_path = f"/data6/shubham/PC/data/masks/subj0{subject}/{functional}.nii.gz"

    if(functional1 == "prf-visualrois_with_high_vis_rgn_enc"): functional_path1 = paths.mask_path.format(subject, "prf-visualrois")
    else: functional_path1 = paths.mask_path.format(subject, functional1)

    if(functional2 == "prf-visualrois_with_high_vis_rgn_enc"): functional_path2 = paths.mask_path.format(subject, "prf-visualrois")
    else: functional_path2 = paths.mask_path.format(subject, functional2)

    functional_mask1 = load_from_nii(functional_path1)
    mask1 = (functional_mask1 == i1)
    size1 = {subject : np.sum(mask1)}
    print(f"num of voxels = {size1}")

    functional_mask2 = load_from_nii(functional_path2)
    mask2 = (functional_mask2 == i2)
    size2 = {subject : np.sum(mask2)}
    print(f"num of voxels = {size2}")
    


    if(functional1 != "prf-visualrois"):
        brain_model1 = encoder_model_vit_update.downproject_CLIP_split_linear_higher(size1)
    else :
        brain_model1 = encoder_model_vit_update.downproject_CLIP_split_linear_early(size1)

    if(functional2 != "prf-visualrois"):
        brain_model2 = encoder_model_vit_update.downproject_CLIP_split_linear_higher(size2)
    else :
        brain_model2 = encoder_model_vit_update.downproject_CLIP_split_linear_early(size2)

    # weights = torch.load(f"/media/internal_8T/prashant/BrainDiVE/results2/checkpoints/subject_{subject}_neurips_split_VIT_last_fully_linear/{functional}/{region}/00100.chkpt")
    # weights = torch.load(f"/data6/shubham/PC/data/results2/checkpoints/subject_{subject}_neurips_split_VIT_last_fully_linear/{functional}/{region}/00080.chkpt")
    weights1 = torch.load(paths.weights_path.format(subject, functional1, region1))
    brain_model1.load_state_dict(weights1["network"], strict=True)

    weights2 = torch.load(paths.weights_path.format(subject, functional2, region2))
    brain_model2.load_state_dict(weights2["network"], strict=True)

    brain_model1.cuda()
    brain_model1.eval()
    for name, param in brain_model1.named_parameters():
        param.requires_grad = False
    assert not brain_model1.training

    brain_model2.cuda()
    brain_model2.eval()
    for name, param in brain_model2.named_parameters():
        param.requires_grad = False
    assert not brain_model2.training

    try: del pipe    
    except: pass
        
    try: del pipe2    
    except: pass
        
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

    repo_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")  # I made this update as the the reccom. by the compiler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe2 = pipe.to("cuda")


    print(f"starting {region1}")

    print("Starting S{}{}".format(subject, region1))

    print(f"starting {region2}")

    print("Starting S{}{}".format(subject, region2))
    
    # region_seeds = list(range(2000 * subject + i * 20, 2000 * subject + (i + 1) * 20))
    i = i1+i2
    region_seeds = list(range(20000 * int(subject) * 2 + i * 20, 20000 * int(subject) * 2 + i * 20 + 5))
    # region_seeds = list(range(2000 * subject + i * 20 + 4, 2000 * subject + i * 20 + 5))

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


    def loss_function_brainDiVE(image_input):
        image_features = backbone(image_input)

        pred_response1 = brain_model1.forward(image_features[0][0], image_features[0][1], image_features[1], [subject]).reshape(-1)
        pred_response2 = brain_model2.forward(image_features[0][0], image_features[0][1], image_features[1], [subject]).reshape(-1)
        
        pred_response11 = torch.mean(pred_response1)
        pred_response21 = torch.mean(pred_response2)

        pred_response = (pred_response11*torch.exp(-pred_response11) * j1 + pred_response21*torch.exp(-pred_response21) * j2)/(torch.exp(-pred_response11) * j1 + torch.exp(-pred_response21) * j2)
        # pred_response = pred_response11 + pred_response21
        
        return -pred_response
    
    j11 = {1: "+", -1: "-"}.get(j1, None)
    j21 = {1: "+", -1: "-"}.get(j2, None)
    # current_folder = f"/data6/shubham/PC/data/results2/BrainDiVE_gen_images/sub_{subject}/{functional}/{region}"

    current_folder = f"/data6/shubham/PC/data/results2/BrainDiVE_gen_images/sub_{subject}/combinations_nonlinear/{functional1}_{functional2}/({j11}){region1}+({j21}){region2}"

    os.makedirs(current_folder, exist_ok=True)
    offset = 0

    pipe.brain_tweak = loss_function_brainDiVE

    for seed in region_seeds:
        print(f"starting {seed}")
        offset += 1
        print("Starting {}".format(str(offset).zfill(5)))
        if offset % 20 == 0:
            print("S{}, {} region, {}/500".format(subject, region1, offset))
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
    parser.add_argument('--subject_id', default="7", type = str)

    parser.add_argument('--functional1', default="prf-visualrois_with_high_vis_rgn_enc", type = str)
    parser.add_argument('--region1', default="V1v", type = str)
    parser.add_argument('--i1', default=1, type = int)
    parser.add_argument('--j1', default=1, type = int)

    parser.add_argument('--functional2', default="prf-visualrois_with_high_vis_rgn_enc", type = str)
    parser.add_argument('--region2', default="hV4", type = str)
    parser.add_argument('--i2', default=7, type = int)
    parser.add_argument('--j2', default=1, type = int)

    parser.add_argument('--gpu', default="3", type = str)
    args = parser.parse_args()
    main(args.subject_id, args.functional1, args.region1, args.i1, args.j1, args.functional2, args.region2, args.i2, args.j2, args.gpu)