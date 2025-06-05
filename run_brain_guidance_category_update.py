import sys
import os

sys.path.append("/media/internal_8T/prashant")
# os.environ["HF_HOME"] = "/ocean/projects/soc220007p/aluo/cache"  #I commented both the line

os.environ["HF_HOME"] = "/media/internal_8T/prashant/cache"

# print(random.getstate())
# exit()
import timm

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

from self_super_reconst import temp_load_enc

def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()

print("Creating CLIP ViT")
# backbone = timm.create_model('vit_base_patch16_clip_224.laion2b', pretrained=True)   #Commented by author

# backbone = model_vit.feature_extractor_vit()
backbone = encoder_model_vit.feature_extractor_vit()  #I made this change from the above line

backbone.eval()
backbone.cuda()


assert not backbone.training
for name, param in backbone.named_parameters():
    param.requires_grad = False
print("Created CLIP ViT and moved to GPU")

# NUM_TO_GENERATE = 1000
NUM_TO_GENERATE = 20

functional = {}


print("Constructing the 3D to valid cortex mask")

# for s in [1,2,3,4,5,6,7,8]:
for s in [1]:
    selected = []

    # for roi_strings in ["prf-visualrois.nii.gz","floc-bodies.nii.gz", "floc-faces.nii.gz", "floc-places.nii.gz", "floc-words.nii.gz", "food", "HCP"]:
    for roi_strings in ["prf-visualrois.nii.gz","floc-bodies.nii.gz", "floc-faces.nii.gz", "floc-places.nii.gz", "floc-words.nii.gz", "HCP"]:
       
        if (not (roi_strings == "food")) and (not (roi_strings == "HCP")):

            # full_path = "/ocean/projects/soc220007p/aluo/rois/subj0{}/{}".format(s, roi_strings)
            full_path = "/media/internal_8T/prashant/BrainDiVE/nsd_data/nsddata/ppdata/subj0{}/func1pt8mm/roi/{}".format(str(s), roi_strings)  # i made this change from above line

            all_roi = load_from_nii(full_path).astype(np.single)
            selected.append(all_roi>=0.5)


        elif roi_strings == "food":
            mask = np.load("/ocean/projects/soc220007p/aluo/subj_{}_food_mask.npy".format(s))  #not found equivalently

            # mask2 = load_from_nii("/ocean/projects/soc220007p/aluo/rois/subj0{}/nsdgeneral.nii.gz".format(s))
            mask2 = load_from_nii("/media/internal_8T/BrainDiVE/nsd_data/nsddata/ppdata/subj0{}/func1pt8mm/roi/nsdgeneral.nii.gz".format(s))

            # Construct flat mask (cortex voxels)
            cortex_mask = mask2[mask2>-0.5]
            container = np.zeros(cortex_mask.shape)
            container[mask] = 1.0

            # Construct the 3D mask, then fill in the flat voxels
            original_shape = np.zeros(mask2.shape)
            original_shape[mask2>-0.5] = container
            selected.append(original_shape>=0.5)


        elif roi_strings == "HCP":

            # hcp_mask = np.load("/ocean/projects/soc220007p/aluo/data/best_HCP.npy")
            hcp_mask = np.load("/media/internal_8T/prashant/BrainDiVE/NSD_zscored/best_HCP.npy")  

            # nsdgeneral = load_from_nii("/ocean/projects/soc220007p/aluo/rois/subj0{}/nsdgeneral.nii.gz".format(s))
            nsdgeneral = load_from_nii("/media/internal_8T/prashant/BrainDiVE/nsd_data/nsddata/ppdata/subj0{}/func1pt8mm/roi/nsdgeneral.nii.gz".format(s))

            container = np.zeros_like(nsdgeneral)
            
            # full_path = "/ocean/projects/soc220007p/aluo/rois/subj0{}/{}_MMP1.nii.gz".format(s, roi_strings)  #found in my directory: /media/internal_8T/prashant/BrainDiVE/nsd_data/nsddata/ppdata/subj07/func1pt8mm/roi/HCP_MMP1.nii.gz
            full_path = "/media/internal_8T/prashant/BrainDiVE/nsd_data/nsddata/ppdata/subj0{}/func1pt8mm/roi/{}_MMP1.nii.gz".format(s, roi_strings)
            
            all_roi = load_from_nii(full_path).astype(np.int32)
            for i in hcp_mask[:45]:
                container[all_roi==i] += 1.0
            selected.append(container>=0.5)

    # functional.append(np.logical_or.reduce(selected))
    functional[s] = np.logical_or.reduce(selected)  #I made this change from the above line

print("Completed loading of all subjects masks")
import random
from base64 import b64encode
random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))

all_subjects = [1]
random.shuffle(all_subjects)
# We shuffle here so the slurm run tries to avoid collisions
# ["RSC", "PPA", "OPA", "FFA", "OFA"]
experiment_id = {"bodies":0, "faces":1, "places":2, "words":3, "food":4,
                 "RSC":5, "PPA":6, "OPA":7, "FFA":8, "OFA":9}

# subject = 1
# TODO remove
for subject in all_subjects:
    # subject = 2
    # TODO remove


# *****************************************************************************************************
    # with open("/ocean/projects/soc220007p/aluo/DiffusionInception/random_seeds.pkl", "rb") as fff:
    #     all_subject_seeds = pickle.load(fff)
    # subject_seeds = all_subject_seeds[subject]

# *****************************************************************************************************


    print("Starting subject {}".format(str(subject)))
    try:
        del dataset
    except:
        pass


    try:
        del brain_model
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()


    class myarg():
        def __init__(self):
            self.subject_id = [str(subject)]

            # self.neural_activity_path = "/ocean/projects/soc220007p/aluo/data/cortex_subj_{}.npy"
            self.neural_activity_path = "/media/internal_8T/prashant/BrainDiVE/NSD_zscored/cortex_subj_{}.npy"


            # self.image_path = "/ocean/projects/soc220007p/aluo/data/image_data.h5py"
            self.image_path = "/media/internal_8T/prashant/BrainDiVE/image_data.h5py"


            self.double_mask_path = "/ocean/projects/soc220007p/aluo/double_mask_HCP.pkl"


            self.volume_functional_path = "/ocean/projects/soc220007p/aluo/volume_to_functional.pkl"


            # self.early_visual_path = "/ocean/projects/soc220007p/aluo/rois/subj0{}/prf-visualrois.nii.gz"
            self.early_visual_path = "/media/internal_8T/prashant/BrainDiVE/nsd_data/nsddata/ppdata/subj0{}/func1pt8mm/roi/prf-visualrois.nii.gz"



    other_args = myarg()
    dataset = neural_loader(other_args)
    
    # brain_model = model_vit.downproject_CLIP_split_linear(num_early_output=dataset.early_sizes, num_higher_output=dataset.higher_sizes)
    brain_model = encoder_model_vit.downproject_CLIP_split_linear(num_early_output=dataset.early_sizes, num_higher_output=dataset.higher_sizes)

    weights = torch.load("/media/internal_8T/prashant/BrainDiVE/results/subject_{}_neurips_split_VIT_last_fully_linear/00100.chkpt".format(str(subject)))
    
    # brain_model.load_state_dict(weights["network"], strict=True)
    brain_model.load_state_dict(weights["network"], strict=False)

    brain_model.cuda()
    brain_model.eval()
    for name, param in brain_model.named_parameters():
        param.requires_grad = False
    assert not brain_model.training

    ############################## Some code to map from our early/higher voxels back to cortical voxels (HCP45 + functional)

    # early_full_path = "/ocean/projects/soc220007p/aluo/rois/subj0{}/{}".format(subject, "prf-visualrois.nii.gz")
    early_full_path = "/media/internal_8T/prashant/BrainDiVE/nsd_data/nsddata/ppdata/subj0{}/func1pt8mm/roi/{}".format(subject, "prf-visualrois.nii.gz")

    early_visual = load_from_nii(early_full_path).astype(np.int32)


    # volume_functional_mask = functional[subject - 1]  # 3D bool mask that goes from volume to ROI voxels
    # early_vis_mask = torch.from_numpy((early_visual > 0.5)[volume_functional_mask])
    # higher_vis_mask = torch.from_numpy((early_visual < 0.5)[volume_functional_mask])

    early_vis_mask = dataset.early_visual_mask[str(subject)]
    higher_vis_mask = dataset.higher_visual_mask[str(subject)]



    ######## Where is the memory leak??????
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
    pipe = mypipelineSAG.from_pretrained(repo_id, torch_dtype=torch.float16, variant="fp16")  # I made this update as the the reccom. by the compiler

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe2 = pipe.to("cuda")



    import inspect

    # Assuming you have a pipeline object `pipe` with a scheduler
    scheduler = pipe.scheduler

    # Check if 'add_noise' method exists in the scheduler
    has_add_noise = hasattr(scheduler, 'add_noise')

    # Print the result
    print(f"Does 'pipe.scheduler' have 'add_noise' method? {has_add_noise}")

    # If it exists, inspect its signature
    if has_add_noise:
        add_noise_signature = inspect.signature(scheduler.add_noise)
        print(f"'add_noise' method signature: {add_noise_signature}")



    # regions = ["bodies", "faces", "places", "words", "food"]

    # regions = ["RSC", "PPA", "OPA", "FFA-1", "OFA"]
    regions = ["RSC", "PPA", "OPA", "OFA"]

    random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))
    random.shuffle(regions)
    # shuffle here to avoid slurm collision
    for i, region in enumerate(regions):
        print("Starting S{} {}".format(subject, region))
        # region = "words"
        # region = "OFA"

        #TODO REMOVE

# ****************************************************************************
        # random_seed_idx = experiment_id[region]
        # region_seeds = list(subject_seeds[random_seed_idx][:NUM_TO_GENERATE].copy())
        # random.seed(a=b64encode(os.urandom(5)).decode('utf-8'))
        # random.shuffle(region_seeds)
# ****************************************************************************
#This complete block I commented following line will does the equivalent job

        region_seeds = list(range(2000 * subject + i * 20, 2000 * subject + (i + 1) * 20))



        # roi_name = "floc-{}".format(region)

        try:
            del mask_tensor
        except:
            pass

        try:
            del region_mask_flat
            del region_mask
        except:
            pass

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

        # ["RSC", "PPA", "OPA", "FFA", "OFA"]
        meta_region_list = {"RSC":"places", "PPA":"places", "OPA":"places","FFA":"faces", "OFA":"faces"}
        region_sub_id_list = {"OPA":[1], "PPA":[2], "RSC":[3], "FFA":[2,3], "OFA":[1]}
        meta_region = meta_region_list[region]

        if meta_region in ["bodies", "faces", "places", "words"]:

            # region_mask_string = "/ocean/projects/soc220007p/aluo/refined_roi/{}_S{}_t2.npy".format(meta_region, subject)  #I commented
            # region_mask = np.load(region_mask_string)   #I commented


            roi_strings = "floc-{}.nii.gz".format(meta_region)

            # roi_id_path = "/ocean/projects/soc220007p/aluo/rois/subj0{}/{}".format(subject, roi_strings)
            roi_id_path = f"/media/internal_8T/prashant/BrainDiVE/nsd_data/nsddata/ppdata/subj0{subject}/func1pt8mm/roi/{roi_strings}"  #made this change from the above line


            # print(roi_id_path, "PATH #################")
            loaded_roi_ids = load_from_nii(roi_id_path).astype(np.int32)
            # print(loaded_roi_ids.shape, "SHAPE")
            all_sub_masks = []
            for desired_id in region_sub_id_list[region]:
                all_sub_masks.append(loaded_roi_ids == desired_id)
            all_sub_masks = np.logical_or.reduce(all_sub_masks)
         

            # region_mask = np.logical_and(region_mask, all_sub_masks)
            region_mask = all_sub_masks

            # print(np.sum(region_mask), meta_region, region, subject, "new region mask")
            # exit()


        elif meta_region in ["food"]:
            region_mask_string = "/ocean/projects/soc220007p/aluo/food_s{}.npy".format(subject)
            region_mask = np.load(region_mask_string)>0.5
        else:
            print("mistake!")
        # print(region, region_mask.shape, region_mask.dtype, np.unique(region_mask))
        # exit()
        region_mask_flat = region_mask[functional[subject]]
        mask_tensor = torch.from_numpy(region_mask_flat).bool().to("cuda")
        print(f"shape of mask_tensor = {mask_tensor.shape}")

        enc, num_voxels = temp_load_enc.load_encoder(s, region)
        enc.cuda()
        enc.eval()
        for name, param in enc.named_parameters():
            param.requires_grad = False
        assert not enc.training

        # def loss_function_gaziv(image_input):

        #     image = image_input.squeeze(0)
        #     to_pil_image = transforms.ToPILImage()
        #     pil_image = to_pil_image(image)

        #     all_act = temp_load_enc.get_activations(enc, pil_image, num_voxels)

        #     ans1 = np.mean(all_act)
        #     ans = torch.tensor(ans1, dtype = torch.float32, requires_grad = True, device = 'cuda')
        #     print(f"ans_type = {type(ans)}, ans_shape = {ans.shape}, ans_dtype = {ans.dtype}")
        #     return ans
        

        def loss_function_higher(image_input):
            print(f"image_type = {type(image_input)}, image_shape = {image_input.shape}, image_dtype = {image_input.dtype}")
            image_features = backbone(image_input)
            predicted_voxels_higher = brain_model.forward_higher(image_features[0][0], image_features[0][1], image_features[1], [subject]).reshape(-1)
            pred_response = torch.zeros(higher_vis_mask.shape).float().to("cuda")
            # pred_response[early_vis_mask] = torch.zeros(torch.sum(early_vis_mask)).float().to("cuda")
            pred_response[higher_vis_mask] = predicted_voxels_higher

            # return -torch.mean(pred_response[mask_tensor])
            print(f"pred_response_type = {type(pred_response)}, pred_response_shape = {pred_response.shape}, pred_response_dtype = {pred_response.dtype}")
            ans = torch.mean(pred_response)
            print(f"a_type = {type(ans)}, a_shape = {ans.shape}, a_dtype = {ans.dtype}, a_requires_grad = {ans.requires_grad}, ans_grad_fn = {ans.grad_fn}")

            return -ans
            # return -torch.mean(pred_response) #I made this update from the above line
            

        def loss_function_early(image_input):
            image_features = backbone(image_input)
            predicted_voxels_early = brain_model.forward_early(image_features[0][0], image_features[0][1], image_features[1], [subject]).reshape(-1)
            pred_response = torch.zeros(higher_vis_mask.shape).float().to("cuda")
            pred_response[early_vis_mask] = predicted_voxels_early
            # pred_response[higher_vis_mask] = torch.zeros(torch.sum(higher_vis_mask)).to("cuda")
            return -torch.mean(pred_response[mask_tensor])

        current_folder = "/media/internal_8T/prashant/BrainDiVE/results/gen_images/{}/{}".format("S{}".format(subject), region + "_all_t2")
        # try:
        os.makedirs(current_folder, exist_ok=True)
        offset = 0

        pipe.brain_tweak = loss_function_higher
        # pipe.brain_tweak = loss_function_gaziv  # I modified this to find the loss using Gaziv's encoder

        for seed in region_seeds:
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
            image = pipe("", sag_scale=0.75, guidance_scale=0.0, num_inference_steps=50, generator=g, clip_guidance_scale=130.0)
            if os.path.exists(image_name):
                continue
            image.images[0].save(image_name, format="PNG", compress_level=6)



            