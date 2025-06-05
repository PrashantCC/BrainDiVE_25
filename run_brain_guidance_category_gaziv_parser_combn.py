import sys
import os

sys.path.append("/data6/home2/spshubham/prashant")
# os.environ["HF_HOME"] = "/ocean/projects/soc220007p/aluo/cache"  #I commented both the line

os.environ["HF_HOME"] = "/data6/home2/spshubham/prashant/cache"

# print(random.getstate())
# exit()
import timm
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from encoder_dataloader_update import neural_loader
from brain_guide_pipeline_graph import mypipelineSAG
import pickle
import gc
import encoder_model_vit
import nibabel as nib
import os
import time
import h5py
import paths
from PIL import Image, ImageDraw, ImageFont, ImageOps
import matplotlib.pyplot as plt

from self_super_reconst import temp_load_enc

def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()



def add_text_below_image(image, text, extra_height=100, font_size=40):

    new_image = ImageOps.expand(image, border=(0, 0, 0, extra_height), fill='white')
    draw = ImageDraw.Draw(new_image)

    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)   
    except IOError: font = ImageFont.load_default()

    image_width = new_image.width
    image_height = image.height
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_x = image_width // 2
    text_y = image_height + extra_height // 2

    draw.text((text_x, text_y), text, font=font, fill='black', anchor="mm")

    return new_image


import random
from base64 import b64encode
random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))

# all_subjects = [1, 2, 5, 7]

# for subject in all_subjects:

class Combined_Activation(nn.Module):

    def __init__(self, subject, region1, region2, j1, j2):
        super(Combined_Activation, self).__init__()

        self.j1 = j1
        self.j2 = j2
        self.enc1, self.num_voxels1 = temp_load_enc.load_encoder(subject, region1)
        self.enc2, self.num_voxels2 = temp_load_enc.load_encoder(subject, region2)

    def forward(self, image):
        
        act1 = temp_load_enc.get_activations(self.enc1, image, self.num_voxels1)
        act2 = temp_load_enc.get_activations(self.enc2, image, self.num_voxels2)  
        
        mean_act1 = torch.mean(act1)
        mean_act2 = torch.mean(act2)

        mean_act11 = self.j1 * mean_act1
        mean_act21 = self.j2 * mean_act2

        comb_mean_act = (mean_act11 * torch.exp(-mean_act11) + mean_act21 * torch.exp(-mean_act21)) / (torch.exp(-mean_act11) + torch.exp(-mean_act21))

        return comb_mean_act, mean_act1, mean_act2
    

def main(subject, region1, i1, j1, region2, i2, j2, gpu):

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

    # pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")
    pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16)
    # pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")  # I made this update as the the reccom. by the compiler

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe2 = pipe.to("cuda")
    pipe2.to(device)


    print(f"starting {region1}")
    print("Starting S{} {}".format(subject, region1))
    print(f"starting {region2}")
    print("Starting S{} {}".format(subject, region2))
    

    region_seeds = list(range(30000, 30003))

    
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


    enc1, num_voxels1 = temp_load_enc.load_encoder(subject, region1)

    
    enc1.to(device)
    enc1.eval()

    for name, param in enc1.named_parameters():
        param.requires_grad = False
    assert not enc1.training

    enc2, num_voxels2 = temp_load_enc.load_encoder(subject, region2)

    
    enc2.to(device)
    enc2.eval()

    for name, param in enc2.named_parameters():
        param.requires_grad = False
    assert not enc2.training

    def loss_function_gaziv(image_input):

        image = image_input.squeeze(0)
        # to_pil_image = transforms.ToPILImage()
        # pil_image = to_pil_image(image)

        act1 = temp_load_enc.get_activations(enc1, image, num_voxels1)
        act2 = temp_load_enc.get_activations(enc2, image, num_voxels2)
        # all_act = torch.tensor(all_act1, dtype = torch.float32, requires_grad = True, device = 'cuda')

        mean_act1 = torch.mean(act1)
        mean_act2 = torch.mean(act2)

        mean_act1_dash = mean_act1 * j1
        mean_act2_dash = mean_act2 * j2

        # pred_response = (pred_response11 * torch.exp(-pred_response11) * j1 + pred_response21 * torch.exp(-pred_response21) * j2) / (torch.exp(-pred_response11) + torch.exp(-pred_response21))
        comb_mean_act = mean_act1_dash + mean_act2_dash
        # pred_response = 3*j1*pred_response11 + 4*j2*pred_response21
        return comb_mean_act, mean_act1, mean_act2

    combined_model = Combined_Activation(subject, region1, region2, j1, j2).to(device)

    def loss_function_gaziv_dash(image_input):
        comb_mean_act, mean_act1, mean_act2 = combined_model(image_input)
        return comb_mean_act, mean_act1, mean_act2
        
    
    current_folder = f"/data6/home2/spshubham/prashant/data/results_gaziv_algonauts/joint_encoder_master_images/sub{subject}/multiple_regions/{region1}_{region2}/({j1}){region1}_({j2}){region2}"
    
    os.makedirs(current_folder, exist_ok=True)
    offset = 0

    # pipe.brain_tweak = Combined_Activation(subject, region1, region2, j1, j2)
    pipe.brain_tweak = loss_function_gaziv
    
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
        image, act_vector = pipe("", sag_scale=0.75, guidance_scale=0.0, num_inference_steps=50, generator=g, clip_guidance_scale=130.0)
        if os.path.exists(image_name):
            continue
        new_image = add_text_below_image(image.images[0], f"({j1}){region1} + ({j2}){region2}", extra_height=30, font_size=20)
        new_image.save(image_name, format="PNG", compress_level=6)

    
        # # Create a plot
        # plt.figure(figsize=(10, 6))
        # plt.plot(np.arange(200), act_vector[0], label='Combined activation', color='blue')
        # plt.plot(np.arange(200), act_vector[1], label='Region_1 activation', color='red')
        # plt.plot(np.arange(200), act_vector[2], label='Region_2 activation', color='green')
        # plt.xlabel('time steps')
        # plt.ylabel('activity')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f"{current_folder}/({j1}){region1}_({j2}){region2}_{str(seed).zfill(12)}.png")  # Save as PNG file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "process subject")
    parser.add_argument('--subject_id', default=2, type = int)


    parser.add_argument('--region1', default="FFA-1", type = str)
    parser.add_argument('--i1', default=2, type = int)
    parser.add_argument('--j1', default=-1.0, type = float)

  
    parser.add_argument('--region2', default="PPA", type = str)
    parser.add_argument('--i2', default=2, type = int)
    parser.add_argument('--j2', default=1.0, type = float)

    parser.add_argument('--gpu', default="5", type = str)
    args = parser.parse_args()
    main(args.subject_id, args.region1, args.i1, args.j1, args.region2, args.i2, args.j2, args.gpu)
