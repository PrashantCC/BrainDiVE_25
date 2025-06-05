import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
import pretrainedmodels as pm
import pretrainedmodels.utils as pmutils
from self_super_reconst.utils import *
from self_super_reconst.config_dec import *
from absl import app
# from self_super_reconst.utils.misc import set_gpu

# flags.DEFINE_integer("is_rgbd", 0, '1: RGBD | 2: Depth only')

import pdb

from PIL import Image


def get_voxels(sub, roi):
    cprint1('(*) Loading Voxels')
    subj = format(sub, '02')
    voxels_map = dict(np.load(f'/hdd_home/achin/data/algonauts/algonauts_2023_data/subj{subj}/masked/fmri_roi_shapes.npz'))

    n_voxels = voxels_map[roi][1]

    return n_voxels


def load_encoder(subj, roi, enc_cpt_path):
    # pdb.set_trace()
    cprint1('Loading Encoder')
    # set_gpu()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cprint1('(*) CUDA_VISIBLE_DEVICES: {} ({} workers per GPU)'.format(os.environ['CUDA_VISIBLE_DEVICES'], 5))
    
    n_voxels = get_voxels(subj, roi)

    cprint1('(*) Separable Encoder')
    enc = make_model('SeparableEncoderVGG19ml', n_voxels, 3, drop_rate=0.25)

    ######## ONLY FOR CPU ########

    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # cpu_state_dict = torch.load(enc_cpt_path, map_location="cpu")['state_dict']
    # for k, v in cpu_state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # assert os.path.isfile(enc_cpt_path)
    # print('\t==> Loading checkpoint {}'.format(os.path.basename(enc_cpt_path)))
    # enc.load_state_dict(new_state_dict)

    ######## ONLY FOR CPU ########

    enc.cuda()    # REMEMBER TO CHANGE THIS

    enc = nn.DataParallel(enc)

    assert os.path.isfile(enc_cpt_path)
    print('\t==> Loading checkpoint {}'.format(os.path.basename(enc_cpt_path)))
    enc.load_state_dict(torch.load(enc_cpt_path)['state_dict'])

    enc.eval()

    # print(imgs)

    # pdb.set_trace()

    return enc, n_voxels



def get_activations(enc, imgs, n_voxels):

    gens = len(imgs)//20

    img_arr = np.array(imgs).reshape(gens,20)
    all_act = np.zeros([gens,20,n_voxels])

    img_xfm_basic = transforms.Compose([
                # transforms.Resize(size=(112,112), interpolation=Image.BILINEAR),
                # transforms.CenterCrop((112,112)),

                transforms.Resize(size=(256,256), interpolation=Image.BILINEAR),
                transforms.CenterCrop((256,256)),
                transforms.ToTensor(),
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    
    im_batch = torch.zeros([20,3,256,256])


    from tqdm import tqdm
    for gen in tqdm (range(gens)):
        for repeat in range(20):
            loc = img_arr[gen,repeat]
            im = Image.open(loc).convert('RGB')
            img_tensor = img_xfm_basic(im)

            im = img_tensor
            im_batch[repeat] = im

            # im = im[None, :, :, :]     # im_crop -> im

        # im = im.cuda()


        im_batch = im_batch.cuda()           # REMEMBER TO CHANGE THIS


        fmri = enc(im_batch)

        # all_act[gen,repeat] = fmri[0].cpu().detach().numpy()
        
        all_act[gen] = fmri.cpu().detach().numpy()

    # pdb.set_trace()

    print(all_act.shape)

    return all_act

    

def main(argv):
    # pdb.set_trace()
    del argv

    # subject_num = 6
    subs = [6]
    # roi_key = 'V1v'

    roi_list_gen1 = ['V1v', 'FFA-2', 'PPA']
    roi_list_gen2 = ['V1v', 'FFA-2', 'PPA']
    suppress = True

    prefix = '0908-'

    top_10_class = [379, 528, 851, 917]


    for sub in subs:
        for roi1 in roi_list_gen1:
            for roi2 in roi_list_gen2:
                if roi1 == roi2:
                    continue

                print('ROI1 = ', roi1, 'ROI2 = ', roi2)

                loc1 = f"/hdd_home/achin/projs/ss_recon/SelfSuperReconst/checkpoints/sub{sub}_{roi1}_rgb_only.pth.tar"
                loc2 = f"/hdd_home/achin/projs/ss_recon/SelfSuperReconst/checkpoints/sub{sub}_{roi2}_rgb_only.pth.tar"

                enc1, nvoxels1 = load_encoder(sub, roi1, loc1)
                enc2, nvoxels2 = load_encoder(sub, roi2, loc2)

                if(suppress):
                    roi_name = "enhance_%s_suppress_%s"%(roi1, roi2)
                else:
                    roi_name = "enhance_%s_enhance_%s"%(roi1, roi2)

                for cat_idx, c in enumerate(top_10_class):

                    # folder = "/hdd_home/achin/projs/saten/xavierNew/AllCode/NSD_SOS_Xdream/2205_joint-5-soft/Subject6bg/class379/enhance_FFA-2_suppress_V1v/class_379_sub6_FFA-2-V1v_seed0_trunc1/backup/"
                    
                    folder = "/hdd_home/achin/projs/saten/xavierNew/AllCode/NSD_SOS_Xdream/2405_joint-soft/Subject%d"%sub+"bg/class%s"%c+"/"+roi_name+"/class_%d_sub%d_%s-%s_seed0_trunc1/backup/"%(c, sub, roi1, roi2)

                    # "/hdd_home/achin/projs/saten/xavierNew/AllCode/NSD_SOS_Xdream/2205_joint/Subject6bg/class379/enhance_V1v_suppress_FFA-2/class_379_sub6_V1v-FFA-2_seed0_trunc1/backup/"


                    imgs = []

                    for filename in os.listdir(folder):
                        if filename.startswith("block"): 
                            file = os.path.join(folder, filename)
                            imgs.append(file)
                        else:
                            continue



                    roi1_act = get_activations(enc1, imgs, nvoxels1)

                    roi2_act = get_activations(enc2, imgs, nvoxels2)

                    roi1_act = roi1_act.mean(axis=2)
                    roi2_act = roi2_act.mean(axis=2)

                    if suppress == True:
                        factor = -1
                    else:
                        factor = 1

                    Z1_final = roi1_act
                    Z2_final = roi2_act*factor
                    
                    # import pdb; pdb.set_trace()

                    #### implement the exponential softmin type function
                    Z = (Z1_final*np.exp(-Z1_final) + Z2_final*np.exp(-Z2_final))/(np.exp(-Z1_final) + np.exp(-Z2_final))

                    name = str(roi_name)

                    name = prefix + 'all_activations/S%d/'%sub + name + '/class_%d/'%(c)
                    if not os.path.exists(name):
                        os.makedirs(name)

                    np.save(name + 'class_%d_'%c + roi1 + '.npy', roi1_act)
                    np.save(name + 'class_%d_'%c + roi2 + '.npy', roi2_act)
                    np.save(name + 'class_%d_'%c + 'Z.npy', Z)



if __name__ == '__main__':
    app.run(main)



# subject_num = 1
# roi_key = 'V1v'
# loc = f"/hdd_home/achin/projs/ss_recon/SelfSuperReconst/checkpoints/sub{subject_num}_{roi_key}_rgb_only.pth.tar"

