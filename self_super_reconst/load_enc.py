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

import glob
from tqdm import tqdm


from PIL import Image


def get_voxels(sub, roi):
    cprint1('(*) Loading Voxels')
    subj = format(sub, '02')
    voxels_map = dict(np.load(f'/hdd_home/achin/data/algonauts/algonauts_2023_data/subj{subj}/masked_new/fmri_roi_shapes.npz'))

    n_voxels = voxels_map[roi][1]

    return n_voxels


def load_encoder(subj, roi, enc_cpt_path):
    # pdb.set_trace()
    cprint1('Loading Encoder')
    # set_gpu()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cprint1('(*) CUDA_VISIBLE_DEVICES: {} ({} workers per GPU)'.format(os.environ['CUDA_VISIBLE_DEVICES'], 5))
    
    n_voxels = get_voxels(subj, roi)

    if(n_voxels == 0):
        return None

    cprint1('(*) Separable Encoder')
    enc = make_model('SeparableEncoderVGG19ml', n_voxels, 3, drop_rate=0.25)

    enc.cuda()

    enc = nn.DataParallel(enc)

    assert os.path.isfile(enc_cpt_path)
    print('\t==> Loading checkpoint {}'.format(os.path.basename(enc_cpt_path)))
    enc.load_state_dict(torch.load(enc_cpt_path)['state_dict'])

    enc.eval()

    # loc = "/hdd_home/achin/projs/saten/xavierNew/AllCode/SOS_Xdream/0305-10_results_achin_fwrf_custom_expt/Subject4bg/class528/class_528_sub4_V1_seed0_trunc1/backup/block049_019_gen049_000994.png"
    # loc = "/hdd_home/achin/projs/saten/xavierNew/AllCode/SOS_Xdream/0305-10_results_achin_fwrf_custom_expt/Subject4bg/class528/class_528_sub4_FFA_seed0_trunc1/backup/block049_005_gen049_000993.png"
    # loc = "/hdd_home/achin/projs/neurogen/NeuroGen/2104_custom_img/S6/ROI02/C0528_repeat11.png"

    # folder = "/hdd_home/achin/projs/saten/xavierNew/AllCode/SOS_Xdream/0305-10_results_achin_fwrf_custom_expt/Subject4bg/class528/class_528_sub4_V1_seed0_trunc1/backup/"
    # folder = "/hdd_home/achin/projs/saten/xavierNew/AllCode/SOS_Xdream/0305-10_results_achin_fwrf_custom_expt/Subject6bg/class528/class_528_sub6_V1_seed0_trunc1/backup/"

    folder = "/hdd_home/achin/data/algonauts/algonauts_2023_data/subj%02d/test_split/test_images/"%subj

    imgs = glob.glob(folder + "*.png")
    imgs.sort()

    fmri_compiled = np.zeros((len(imgs), n_voxels))

    for i, loc in tqdm(enumerate(imgs), total=len(imgs)):
        im = Image.open(loc).convert('RGB')
        img_xfm_basic = transforms.Compose([
            transforms.Resize(size=(256,256), interpolation=Image.BILINEAR),
            transforms.CenterCrop((256,256)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        img_tensor = img_xfm_basic(im)

        im = img_tensor

        # im = im[:3,:,:]/255	
        im = im[None, :, :, :]     # im_crop -> im

        im = im.cuda()

        fmri = enc(im)

        fmri = fmri[0]

        fmri = fmri.detach().cpu().numpy()
        
        fmri_compiled[i,:] = fmri

    if(not os.path.exists("test_individual_fmri_algo/subj%d"%subj)):
        os.mkdir("test_individual_fmri_algo/subj%d"%subj)

    test_fmri_loc = "test_individual_fmri_algo/subj%d/%s.npy" % (subj, roi)

    np.save(test_fmri_loc, fmri_compiled)

    # mean_fmri = np.mean(fmri)

    # pdb.set_trace()

    # return mean_fmri




def main(argv):
    # pdb.set_trace()
    del argv

    # subject_num = 6
    # roi_key = 'FFA-2'

    roi_list = ["V1v", "V1d", "V2v", "V2d", "V3v", "V3d", "hV4", "EBA", "FBA-1", "FBA-2", "mTL-bodies", "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces", "OPA", "PPA", "RSC", "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"]


    for subject_num in [1, 2, 3, 4, 5, 6, 7, 8]:
        for roi_key in roi_list:

            print("#"*50)
            print("Subject %d, ROI %s" % (subject_num, roi_key))

            loc = f"/hdd_home/achin/projs/ss_recon/SelfSuperReconst/checkpoints/sub{subject_num}_{roi_key}_rgb_only.pth.tar"
            load_encoder(subject_num, roi_key, loc)



if __name__ == '__main__':
    app.run(main)



# subject_num = 1
# roi_key = 'V1v'
# loc = f"/hdd_home/achin/projs/ss_recon/SelfSuperReconst/checkpoints/sub{subject_num}_{roi_key}_rgb_only.pth.tar"

