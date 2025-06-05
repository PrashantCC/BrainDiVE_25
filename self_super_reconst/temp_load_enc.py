import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import pretrainedmodels as pm
import pretrainedmodels.utils as pmutils




# from self_super_reconst.utils import *
from .utils import *

# from self_super_reconst.config_dec import *
from .config_dec import *


from absl import app

import numpy as np
import os
# from self_super_reconst.utils.misc import set_gpu

# flags.DEFINE_integer("is_rgbd", 0, '1: RGBD | 2: Depth only')

import sys

# sys.path.append("/hdd_home/achin/projs/saten/xavierNew/AllCode/NSD_SOS_Xdream/xdream_joint/net_utils/")
# sys.path.append("/home/prashant/BrainDiVE/xdream/net_utils/")

sys.path.append("/data6/home2/spshubham/prashant/scripts/BrainDiVE/xdream/net_utils/")


from resnet_load_classifier import load_resnet_18, load_max_vit

from nsd_utils.models import make_model



# import pdb; pdb.set_trace()

from PIL import Image


def get_voxels(sub, roi):
    cprint1('(*) Loading Voxels')
    subj = format(sub, '02')

    # voxels_map = dict(np.load(f'/hdd_home/achin/data/algonauts/algonauts_2023_data/subj{subj}/masked_new/fmri_roi_shapes.npz'))
    voxels_map = dict(np.load(f'/data6/home2/spshubham/prashant/data/gaziv_encoder/algonauts_2023_data/subj0{sub}/masked_new/fmri_roi_shapes.npz'))
    
    n_voxels = voxels_map[roi][1]

    return n_voxels
    


# def load_encoder(subj, roi, enc_cpt_path):
# def load_encoder(subj, roi):
#     # pdb.set_trace()
#     cprint1('Loading Encoder')
#     # set_gpu()

    
#     print("PyTorch version:", torch.__version__)
#     print("CUDA available:", torch.cuda.is_available())
#     if torch.cuda.is_available():	
#         print("CUDA version:", torch.version.cuda)
#         print("CUDA device count:", torch.cuda.device_count())
#         print("Current CUDA device:", torch.cuda.current_device())
#         print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

#     os.environ['CUDA_VISIBLE_DEVICES'] = '3'
#     cprint1('(*) CUDA_VISIBLE_DEVICES: {} ({} workers per GPU)'.format(os.environ['CUDA_VISIBLE_DEVICES'], 5))
    
#     n_voxels = get_voxels(subj, roi)

#     cprint1('(*) Separable Encoder')
#     enc = make_model('SeparableEncoderVGG19ml', n_voxels, 3, drop_rate=0.25)

#     ######## ONLY FOR CPU ########

#     # from collections import OrderedDict
#     # new_state_dict = OrderedDict()
#     # cpu_state_dict = torch.load(enc_cpt_path, map_location="cpu")['state_dict']
#     # for k, v in cpu_state_dict.items():
#     #     name = k[7:]
#     #     new_state_dict[name] = v
#     # assert os.path.isfile(enc_cpt_path)
#     # print('\t==> Loading checkpoint {}'.format(os.path.basename(enc_cpt_path)))
#     # enc.load_state_dict(new_state_dict)

#     ######## ONLY FOR CPU ########


    
#     enc.cuda()    # REMEMBER TO CHANGE THIS
#     enc = nn.DataParallel(enc)

#     enc_cpt_path = f"/data6/shubham/PC/data/gaziv_encoder/checkpoints/sub{subj}_{roi}_rgb_only_best_corr.pth.tar"

#     assert os.path.isfile(enc_cpt_path)
#     print('\t==> Loading checkpoint {}'.format(os.path.basename(enc_cpt_path)))
#     enc.load_state_dict(torch.load(enc_cpt_path)['state_dict'])

#     enc.eval()

#     # print(imgs)

#     # pdb.set_trace()

#     return enc, n_voxels

def load_encoder(subj, roi):
    # pdb.set_trace()
    cprint1('Loading Encoder')
    
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():	
        print("CUDA version:", torch.version.cuda)
        print("CUDA device count:", torch.cuda.device_count())
        print("Current CUDA device:", torch.cuda.current_device())
        print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Since this is for CPU, you don't need to set CUDA_VISIBLE_DEVICES
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    # cprint1('(*) CUDA_VISIBLE_DEVICES: {} ({} workers per GPU)'.format(os.environ['CUDA_VISIBLE_DEVICES'], 5))
    
    n_voxels = get_voxels(subj, roi)

    cprint1('(*) Separable Encoder')
    enc = make_model('SeparableEncoderVGG19ml', n_voxels, 3, drop_rate=0.25)

    ######## ONLY FOR CPU ########

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    enc_cpt_path = f"/data6/home2/spshubham/prashant/data/gaziv_encoder/checkpoints/sub{subj}_{roi}_rgb_only_best_corr.pth.tar"
    
    assert os.path.isfile(enc_cpt_path)
    print('\t==> Loading checkpoint {}'.format(os.path.basename(enc_cpt_path)))
    cpu_state_dict = torch.load(enc_cpt_path, map_location="cpu")['state_dict']
    
    for k, v in cpu_state_dict.items():
        name = k[7:]  # Remove the 'module.' prefix
        new_state_dict[name] = v

    enc.load_state_dict(new_state_dict)

    ######## ONLY FOR CPU ########

    enc.eval()

    return enc, n_voxels





def transform_image(img):
    preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    preprocess_max_vit = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize(256),
        # transforms.ToTensor(),
        # transforms.CenterCrop(150),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # import pdb; pdb.set_trace()

    resnet_im = img
    resnet_im = preprocess_max_vit(resnet_im)
    resnet_im = resnet_im.unsqueeze(0)
    return resnet_im


# def get_classifier_score(img, cat):
#     #### Getting classificaiton score for the image using resnet-18 ####
#     # classification_model = load_resnet_18()

#     classification_model = load_max_vit()

#     # resnet_im = im.copy()
#     # import pdb; pdb.set_trace()

#     #### Applying renet transformations ####
    
#     resnet_im = img.cuda()
    
#     #### Getting the classification score ####
#     classifier_score = classification_model(resnet_im)

#     #### Adding the classification score to the fmri score ####

#     ## Converting to probit score ##
#     classifier_score = torch.nn.functional.softmax(classifier_score, dim=1) 
#     classifier_score = classifier_score.cpu().detach().numpy()

#     classifier_score = classifier_score[:, cat]

#     return classifier_score



# def get_activations(enc, imgs, n_voxels):

#     gens = len(imgs)//20

#     img_arr = np.array(imgs).reshape(gens,20)
#     all_act = np.zeros([gens,20,n_voxels])

#     img_xfm_basic = transforms.Compose([
#                 # transforms.Resize(size=(112,112), interpolation=Image.BILINEAR),
#                 # transforms.CenterCrop((112,112)),

#                 transforms.Resize(size=(256,256), interpolation=Image.BILINEAR),
#                 transforms.CenterCrop((256,256)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                 ])
    
#     im_batch = torch.zeros([20,3,256,256])


#     from tqdm import tqdm
#     for gen in tqdm (range(gens)):
#         for repeat in range(20):
#             loc = img_arr[gen,repeat]
#             im = Image.open(loc).convert('RGB')
#             img_tensor = img_xfm_basic(im)

#             im = img_tensor
#             im_batch[repeat] = im

#             # im = im[None, :, :, :]     # im_crop -> im

#         # im = im.cuda()


#         im_batch = im_batch.cuda()           # REMEMBER TO CHANGE THIS


#         fmri = enc(im_batch)

#         # all_act[gen,repeat] = fmri[0].cpu().detach().numpy()
        
#         all_act[gen] = fmri.cpu().detach().numpy()

#     # pdb.set_trace()

#     print(all_act.shape)

#     return all_act


def get_activations(enc, img, n_voxels):

    img_xfm_basic = transforms.Compose([
        transforms.Resize(size=(256, 256), interpolation=Image.BILINEAR),
        transforms.CenterCrop((256, 256)),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = img_xfm_basic(img)

    img_tensor.requires_grad_()

    im_batch = torch.zeros([1, 3, 256, 256])

    im_batch[0] = img_tensor

    # Move the batch to GPU
    im_batch = im_batch.cuda()

    # Get fMRI activations
    fmri = enc(im_batch)

    # Convert the activations to numpy array
    all_act = fmri #.cpu().detach().numpy()
    all_act = all_act.flatten()

    # print(all_act.shape)

    return all_act


def get_activations_dash(enc, img):

    # img_xfm_basic = transforms.Compose([
    #     transforms.Resize(size=(256, 256), interpolation=Image.BILINEAR),
    #     transforms.CenterCrop((256, 256)),
    #     # transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    
    # img_tensor = img_xfm_basic(img)

    # img_tensor.requires_grad_()

    # im_batch = torch.zeros([1, 3, 256, 256])

    # im_batch[0] = img_tensor

    # im_batch = im_batch.cuda()

    # Get fMRI activations
    fmri = enc(img)

    # Convert the activations to numpy array
    all_act = fmri #.cpu().detach().numpy()

    # print(all_act.shape)

    return all_act


def main(argv):
    # import pdb; pdb.set_trace()
    del argv

    # subject_num = 6
    # subs = [2, 5]
    # subs = [2]
    subs = [2, 7]

    last_gen = 49

    # roi_key = 'V1v'

    # roi_list_gen1 = ['V1v', 'FFA-2', 'PPA']
    # roi_list_gen2 = ['V1v', 'FFA-2', 'PPA']
    # suppress = True

    roi_list = ['V1v', 'hV4']
    # roi_list = ['V1d']

    code = 2

    # roi_list = ['EBA']
    # roi_list = ['FFA-2', 'PPA']

    # roi_list = ['V1v', "FFA-2", "PPA", "hV4", "V2v", "V3v", "VWFA-1", "OWFA", "mfs-words"]
   

    # prefix = 'mar25_nopt_supp_1_class_1_rgb_0_code%d_finalized_imgs_present-'%code
    # prefix = 'mar25_optimal_class_const_v1d_Code%d-'%code
    prefix = 'mar27_opt_code%d_finalized_imgs_present-'%code

    # top_10_class = [379, 388, 528, 531, 637, 772, 851, 917]
    # top_10_class = [379, 528]
    # top_10_class = [386, 917]

    # top_10_class = [379, 386, 528, 772, 917]

    # top_10_class = [15, 50, 355, 609, 736, 839]

    top_10_class = [379, 528]

    # pdb.set_trace()

    for sub in subs:
        for roi in roi_list:

            print('ROI = ', roi)

            loc1 = f"/hdd_home/achin/projs/ss_recon/SelfSuperReconst/checkpoints_algo_30_epochs/sub{sub}_{roi}_rgb_only.pth.tar"
            # loc2 = f"/hdd_home/achin/projs/ss_recon/SelfSuperReconst/checkpoints/sub{sub}_{roi2}_rgb_only.pth.tar"

            enc1, nvoxels1 = load_encoder(sub, roi, loc1)
            # enc2, nvoxels2 = load_encoder(sub, roi2, loc2)

            # if(suppress):
            #     roi_name = "enhance_%s_suppress_%s"%(roi1, roi2)
            # else:
            #     roi_name = "enhance_%s_enhance_%s"%(roi1, roi2)

            for cat_idx, c in enumerate(top_10_class):

                # folder = "/hdd_home/achin/projs/saten/xavierNew/AllCode/NSD_SOS_Xdream/2205_joint-5-soft/Subject6bg/class379/enhance_FFA-2_suppress_V1v/class_379_sub6_FFA-2-V1v_seed0_trunc1/backup/"
                
                "/hdd_home/achin/projs/saten/xavierNew/AllCode/NSD_SOS_Xdream/0408_classifier_only_achin/Subject2bg/class379/"
                # folder = "/hdd_home/achin/projs/saten/xavierNew/AllCode/finalized_imgs_present/control/supp_1_class_1_rgb_0/code%d/Subject%d"%(code, sub)+"bg/class%s"%c+"/class_%d_sub%d_%s_seed0_trunc1/backup/"%(c, sub, roi)

                # folder = "/hdd_home/achin/projs/saten/xavierNew/AllCode/finalized_imgs_present/control/supp_0_class_0_rgb_1/code%d/Subject%d"%(code, sub)+"bg/class%s"%c+"/class_%d_sub%d_%s_seed0_trunc1/backup/"%(c, sub, roi)
                folder = "/hdd_home/achin/projs/saten/xavierNew/AllCode/finalized_imgs_present/optimal/code%d/Subject%d"%(code, sub)+"bg/class%s"%c+"/class_%d_sub%d_%s_seed0_trunc1/backup/"%(c, sub, roi)
                
                # folder = "/hdd_home/achin/projs/saten/xavierNew/AllCode/Corrected_VGG_Class_Xdream/mar25_optimal_class_const_v1d_Code%d_MaxVit_Initnorm_awp_with_grad_1000_steps_1e-2_1e-2/Subject%d"%(code, sub)+"bg/class%s"%c+"/class_%d_sub%d_%s_seed0_trunc1/backup/"%(c, sub, roi)



                imgs = []

                for filename in os.listdir(folder):
                    if filename.startswith("block"): 
                        file = os.path.join(folder, filename)
                        imgs.append(file)
                    else:
                        continue


                ### LAST GEN ONLY ###
                
                all_files = os.listdir(folder)
                all_files.sort()
                last_block_name = ["block%03d"%(last_gen)]
                imgs_last_gen = [s for s in all_files if any(xs in s for xs in last_block_name)][:-1]
                img_list = [folder+s for s in imgs_last_gen]


                # import pdb; pdb.set_trace()

                ###

                # img_list = imgs

                ###
                # import pdb; pdb.set_trace()

                

                roi_act = get_activations(enc1, img_list, nvoxels1)

                # roi2_act = get_activations(enc2, imgs, nvoxels2)

                roi_act = roi_act.mean(axis=2)
                # roi2_act = roi2_act.mean(axis=2)

                # name = str(roi)

                name = prefix + 'all_activations/S%d/'%sub + 'class_%d/'%(c)
                if not os.path.exists(name):
                    os.makedirs(name)

                #save last gen imgs
                ### LAST GEN ONLY ###

                np.save(name + 'class_%d_'%c + roi + '_imgList.npy', np.asarray(imgs_last_gen))
                
                ###

                np.save(name + 'class_%d_'%c + roi + '.npy', roi_act)
                


                ### LAST GEN ONLY ###

                classifier_score = np.zeros((len(imgs_last_gen)//20, 20))
                for gen in tqdm(range(len(imgs_last_gen)//20)):
                    imgs_gen= np.zeros((20, 3, 224, 224))
                    imgs_gen = torch.from_numpy(imgs_gen).float()
                    for idx, img in enumerate(img_list[gen*20:(gen+1)*20]):
                        im = Image.open(img).convert('RGB')
                        im = transform_image(im)
                        imgs_gen[idx] = im

                    # classifier_score.append(get_score(imgs_gen, cat))
                    classifier_score[gen] = get_classifier_score(imgs_gen, c)


                # class_score = get_classifier_score(imgs_last_gen, c)

                np.save(name + 'class_%d_'%c + roi + '_classifier_score.npy', classifier_score)

                ###




if __name__ == '__main__':
    app.run(main)



# subject_num = 1
# roi_key = 'V1v'

# # loc = f"/hdd_home/achin/projs/ss_recon/SelfSuperReconst/checkpoints/sub{subject_num}_{roi_key}_rgb_only.pth.tar"
# loc = f"/troy/prashant/gaziv_encoder/checkpoints/sub{subject_num}_{roi_key}_rgb_only_best_corr.pth.tar"

# enc, num_voxels = load_encoder(subject_num, roi_key, loc)
# print(num_voxels)