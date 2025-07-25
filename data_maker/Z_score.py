import sys
sys.path.append("/data6/shubham/PC/scripts/BrainDiVE/modfit/code")

import os
import numpy as np
import h5py
from scipy.io import loadmat
import PIL.Image
import nibabel as nib
import pandas as pd
import pickle
import time
from utils import default_paths, roi_utils, prf_utils
from model_fitting import initialize_fitting
import gc
import pickle as pkl
import pandas as pd

import imageio
import os
from skimage.transform import rescale, resize
import PIL
from PIL import Image
import h5py

import cortex
# from cortex import config  # Not used anywhere
import numpy as np
# from plotter_utils import volume_maker, visualize_volume, view_1, view_2, make_image, close_handle
import matplotlib.pyplot as plt
import os
os.getcwd()

import nibabel as nib


def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()


nsd_root = default_paths.nsd_root;
stim_root = default_paths.stim_root
beta_root = default_paths.beta_root


# hard coded values based on sessions that are missing for some subs


def get_session_inds_full():
    trials_per_sess = 750
    sess_per_subj = 40
    session_inds = np.repeat(np.arange(0,sess_per_subj), trials_per_sess)
    return session_inds
    

def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()
  
def load_from_mgz(mgz_file):
    return load_from_nii(mgz_file)
  
def load_from_hdf5(hdf5_file, keyname=None):
    data_set = h5py.File(hdf5_file, 'r')
    if keyname is None:
        keyname = list(data_set.keys())[0]
    values = np.copy(data_set[keyname])
    data_set.close()    
    return values

def image_uncolorize_fn(image):
    data = image.astype(np.float32) / 255
    if data.shape[2]==3:
        bw = (0.2126*data[:,:,0:1]+ 0.7152*data[:,:,1:2]+ 0.0722*data[:,:,2:3])
    elif data.shape[1]==3:
        bw = (0.2126*data[:,0:1]+ 0.7152*data[:,1:2]+ 0.0722*data[:,2:3])
    
    return bw

def image_preproc_fn(image):
    data = image.astype(np.float32) / 255
    return data


def get_voxel_mask(subject):
    
    voxel_mask, voxel_index, voxel_roi, voxel_ncsnr, brain_nii_shape = \
                roi_utils.get_voxel_roi_info(subject, which_hemis='concat')
    
    return voxel_mask


def ncsnr_to_nc(ncsnr, average_image_reps=False, subject=None):    
    """
    From Allen (2021) nature neuroscience.    
    """   
    if not average_image_reps: 
        # single trial data
        n = 1
        noise_ceiling = 100 * ncsnr**2 / (ncsnr**2 + 1/n)  
    else:
        if subject is None:
            # assume averaging over three reps of each image
            n = 3
            noise_ceiling = 100 * ncsnr**2 / (ncsnr**2 + 1/n)
        else:
            # for this subject, count how many reps of each image there actually were
            # assume all available sessions are being used
            image_order = get_master_image_order()    
            session_inds = get_session_inds_full()
            max_sess_each_subj = [40,40,32,30,40,32,40,30]
            sessions = np.arange(max_sess_each_subj[subject-1])
            inds2use = np.isin(session_inds, sessions)
            image_order = image_order[inds2use]
            unique, counts = np.unique(image_order, return_counts=True)
            A = np.sum(counts==3)
            B = np.sum(counts==2)
            C = np.sum(counts==1)
            # special version of the NC formula, from nsd data manual
            noise_ceiling = 100 * ncsnr**2 / (ncsnr**2 + (A/3 + B/2 + C/1) / (A+B+C) )
            
    return noise_ceiling
  
def get_nc(subject, average_image_reps=True):
    
    # this is computed in roi_utils.preproc_rois()
    if average_image_reps:
        filename = os.path.join(default_paths.nsd_rois_root, 'S%d_noise_ceiling_avgreps.npy'%subject)
    else:
        filename = os.path.join(default_paths.nsd_rois_root, 'S%d_noise_ceiling_noavg.npy'%subject)
        
    noise_ceiling = np.load(filename)/100
    
    return noise_ceiling

def get_image_data(subject, random_images=False, native=False, npix=240):

    """
    Load the set of NSD images that were shown to a given subject.
    This loads a subject-specific array of images, see [get_subject_specific_images] for details.
    Can also choose to insert random noise instead of the images here.
    """
    
    if random_images==False:        
        print('\nLoading images for subject %d\n'%subject)
        if native:
            image_data = load_from_hdf5(os.path.join(stim_root, 'S%d_stimuli_native.h5py'%subject))     
        else:
            image_data = load_from_hdf5(os.path.join(stim_root, 'S%d_stimuli_%d.h5py'%(subject,npix)))        
    else:        
        print('\nGenerating random gaussian noise images...\n')
        n_images = 10000
        image_data = (np.random.normal(0,1,[n_images, 3, npix,npix])*30+255/2).astype(np.uint8)
        image_data = np.maximum(np.minimum(image_data, 255),0)

    print ('image data size:', image_data.shape, ', dtype:', image_data.dtype, ', value range:',\
        np.min(image_data[0]), np.max(image_data[0]))

    return image_data

def get_master_image_order():    
    """
    Gather the "ordering" information for NSD images.
    masterordering gives zero-indexed ordering of indices (matlab-like to python-like), same for all subjects. 
    consists of 30000 values in the range [0-9999], which provide a list of trials in order. 
    The value in ordering[ii] tells the index into the subject-specific stimulus array that we would need to take to
    get the image for that trial.
    """
    # exp_design_file = "/media/internal_8T/prashant/BrainDiVE/nsd_data/nsddata/experiments/nsd/nsd_expdesign.mat"

    # exp_design_file = os.path.join(nsd_root, 'nsddata','experiments','nsd','nsd_expdesign.mat')
    exp_design_file = "/data6/shubham/PC/data/BrinDiVE_fromPrince/nsd_data/nsddata/experiments/nsd/nsd_expdesign.mat"
    exp_design = loadmat(exp_design_file)
    
    image_order = exp_design['masterordering'].flatten() - 1 
    
    return image_order
      

def load_betas(subject, sessions=[0], voxel_mask=None,  zscore_betas_within_sess=True, volume_space=True):

    """
    Load preprocessed voxel data for an NSD subject (beta weights).
    Always loading the betas with suffix 'fithrf_GLMdenoise_RR.
    Concatenate the values across multiple sessions.
    "sessions" is zero-indexed, add one to get the actual session numbers.
    
    """
    
    if volume_space:
        beta_subj_folder = os.path.join(beta_root, 'subj%02d'%subject, 'func1pt8mm', 'betas_fithrf_GLMdenoise_RR')   
    else:
        beta_subj_folder = os.path.join(beta_root, 'subj%02d'%subject, 'nativesurface', 'betas_fithrf_GLMdenoise_RR')   

    print('Data is located in: %s...'%beta_subj_folder)
    max_sess_each_subj = [40,40,32,30,40,32,40,30]
    if np.any((np.array(sessions)+1)>max_sess_each_subj[subject-1]):
        print('attempting to load sessions:')
        print(sessions+1)
        raise ValueError('trying to load sessions that do not exist for subject %d, only has up to session %d'\
                         %(subject, max_sess_each_subj[subject-1]))
    trials_per_sess = 750
    sess_per_subj = 40    
    n_trials = len(sessions)*trials_per_sess

    for ss, se in enumerate(sessions):
        try:
            del betas
        except:
            pass
        
        try:
            del values
        except:
            pass
        
        gc.collect()
        if volume_space:

            # Load volume space nifti
            fn2load = os.path.join(beta_subj_folder, 'betas_session%02d.nii.gz'%(se+1))
            print('Loading from %s...'%fn2load)
            values = load_from_nii(fn2load).transpose((3,0,1,2))
            print('Raw data:')
            print(values.dtype, np.min(values), np.max(values), values.shape)

            betas = values.reshape((len(values), -1), order='C')

        else:
            # Surface space, concatenate the two hemispheres
            # Must be left then right to match ROI definitions.
            fn2load1 = os.path.join(beta_subj_folder, 'lh.betas_session%02d.hdf5'%(se+1))
            fn2load2 = os.path.join(beta_subj_folder, 'rh.betas_session%02d.hdf5'%(se+1))

            print('Loading from %s...'%fn2load1)        
            values1 = load_from_hdf5(fn2load1)
            print('Raw data:')
            print(values1.dtype, np.min(values1), np.max(values1), values1.shape)

            print('Loading from %s...'%fn2load2)        
            values2 = load_from_hdf5(fn2load2)
            print('Raw data:')
            print(values2.dtype, np.min(values2), np.max(values2), values2.shape)

            betas = np.concatenate((values1, values2), axis=1)

        # divide by 300 to convert back to percent signal change
        betas = betas.astype(np.float32) / 300

        print('Adjusted data (divided by 300):')
        print(betas.dtype, np.min(betas), np.max(betas), betas.shape)
        
        if voxel_mask is not None:        
            betas = betas[:,voxel_mask]

        if zscore_betas_within_sess: 
            print('z-scoring beta weights within this session...')
            mb = np.mean(betas, axis=0, keepdims=True)
            sb = np.std(betas, axis=0, keepdims=True)
            betas = np.nan_to_num((betas - mb) / (sb + 1e-6))
            print ("mean = %.3f, sigma = %.3f" % (np.mean(mb), np.mean(sb)))
            del mb
            del sb
            gc.collect()

        if ss==0:        
            n_vox = betas.shape[1]
            betas_full = np.zeros((n_trials, n_vox), dtype=np.single)

        betas_full[ss*trials_per_sess : (ss+1)*trials_per_sess, :] = betas
        
    return betas_full
       
def get_concat_betas(subject, debug=False):
    
    print('\nProcessing subject %d\n'%subject)
    voxel_mask = get_voxel_mask(subject)
        
    if debug:
        sessions = [0]
    else:
        max_sess_each_subj = [40,40,32,30,40,32,40,30]
        sessions = np.arange(max_sess_each_subj[subject-1])
    zscore_betas_within_sess = True
    volume_space=True
    voxel_data = load_betas(subject, sessions, voxel_mask=voxel_mask, \
                                      zscore_betas_within_sess=zscore_betas_within_sess, \
                                      volume_space=volume_space)
    print('\nSize of full data set [nTrials x nVoxels] is:')
    print(voxel_data.shape)

    save_fn = os.path.join(default_paths.nsd_data_concat_root, 'S%d_allsess_concat_visual.h5py'%subject)

    t = time.time()
    print('saving file to %s'%save_fn)
    with h5py.File(save_fn, 'w') as data_set:
        dset = data_set.create_dataset("betas", np.shape(voxel_data), dtype=np.float64)
        data_set['/betas'][:,:] = voxel_data
        data_set.close()  
    elapsed = time.time() - t
    print('took %.5f sec'%elapsed)
    
def average_image_repetitions(voxel_data, image_order):
    
    n_trials = voxel_data.shape[0]
    n_voxels = voxel_data.shape[1]

    unique_ims = np.unique(image_order)
    print(unique_ims)
    # assert False  #I commented this
    n_unique_ims = len(unique_ims)
    avg_dat_each_image = np.zeros((n_unique_ims, n_voxels), dtype=np.single)
    for uu, im in enumerate(unique_ims):
        inds = image_order==im;
        if np.sum(inds)<3:
            print(im)
        avg_dat_each_image[uu,:] = np.mean(voxel_data[inds,:], axis=0)
        
    return avg_dat_each_image, unique_ims
    
def get_data_splits(subject, sessions=[0], voxel_mask=None, \
                    zscore_betas_within_sess=True, volume_space=True, \
                    average_image_reps = False, \
                    shuffle_images=False, random_voxel_data=False):

    """
    Gather voxel data and the indices of training/testing images, for one NSD subject.
    Not actually loading images here, because all image features are pre-computed. 
    Always leaving out the "shared1000" image subset as my validation set, and training within the rest of the data.
    Can specify a list of sessions to work with (don't have to be contiguous).
    Can specify whether to work in volume or surface space (set volume_space to True or False).
    Can also choose to shuffle images or generate random voxel data at this stage if desired.
    """

    # Load the experiment design file that defines full image order over 30,000 trials
    image_order = get_master_image_order()     # list of 30k images shown to the subject
    
    # Decide which sessions to work with here
    session_inds = get_session_inds_full()   # = [0, ..., 0, 1, ..., 1, ................., 39, ..., 39], len = 750 * 40 = 30k
    
#     print(session_inds, session_inds.shape)
#     print(np.unique(session_inds, return_counts=True))
#     assert False
    if np.isscalar(sessions):
        sessions = [sessions]
    sessions = np.array(sessions)       # ex: subj = 4, max_sess_per_sub[4-1] = 30, sessions = [0, 1, 2, ...., 39]
    max_sess_each_subj = [40,40,32,30,40,32,40,30]
    if np.any((sessions+1)>max_sess_each_subj[subject-1]):
        # adjust the session list that was entered, if the subject is missing some sessions.
        # will alter the list for both images and voxel data.
        print('subject %d only has up to session %d, will load these sessions:'%\
              (subject, max_sess_each_subj[subject-1]))
        sessions = sessions[(sessions+1) <= max_sess_each_subj[subject-1]]  # sessions = [0, 1, 2, ...., 29]
        print(sessions+1)
        assert(len(sessions) > 0)
        
    # session_inds = [0(720), 1(720), ..., 39(720)]
    inds2use = np.isin(session_inds, sessions)   # inds2use = [True(750*30), False(30k - 750*30)], if the no. of sessions for a subj = 30
    session_inds = session_inds[inds2use] # session_inds = [0(750), 1(750), ..., 29(750)]
    image_order = image_order[inds2use]     # list containing images Id's of first 750*30 images
    
    unique_ims = np.unique(image_order)
    print(unique_ims)
    n_unique_ims = len(unique_ims)
    for uu, im in enumerate(unique_ims):
        inds = image_order==im;
        if np.sum(inds)<3:
            print(np.sum(inds), uu)        

    # assert False  I commented this one
    # Now load voxel data (preprocessed beta weights for each trial)
    print('Loading data for sessions:')
    print(sessions+1)
    if not random_voxel_data:
        voxel_data = load_betas(subject, sessions, voxel_mask=voxel_mask, \
                            zscore_betas_within_sess=zscore_betas_within_sess, \
                            volume_space=volume_space)
    else:
        print('creating random normally distributed data instead of loading real data')
        if voxel_mask is not None:
            n_voxels = np.sum(voxel_mask)
        else:
            n_voxels = 10000
        voxel_data = np.random.normal(0,1,(len(image_order), n_voxels))
    print(voxel_data.shape, "Voxel data shape")
    print('\nSize of full data set [n_trials x n_voxels] is:')
    print(voxel_data.shape)
    assert(voxel_data.shape[0]==len(image_order))
    
    # average over repetitions of same image, if desired
    if average_image_reps:
        avg_dat_each_image, unique_ims = average_image_repetitions(voxel_data, image_order)
        voxel_data = avg_dat_each_image # use average data going forward
        image_order = unique_ims # now the unique image indices become new image order
        # NOTE that the unique images can be fewer than 10,000 if the subject
        # is missing some data, or if we are working w just a few sessions. 
        print('\nAfter averaging - size of full data set [n_images x n_voxels] is:')
        print(voxel_data.shape)
        # can't have session inds here because betas are averaged over multiple sessions
        session_inds=None
        
   
    # Get indices to split into training/validation set now
    subj_df = get_subj_df(subject)
    is_shared_image = np.array(subj_df['shared1000'])
    shared_1000_inds = is_shared_image[image_order]
    val_inds = shared_1000_inds
    
    is_trn, is_holdout, is_val = load_image_data_partitions(subject)
    is_trn = is_trn[image_order]
    is_val = is_val[image_order]
    is_holdout = is_holdout[image_order]
    assert(np.all(is_val==val_inds))
    holdout_inds = is_holdout
    
    if shuffle_images:
        print('\nShuffling image order')
        # shuffle each data partition separately
        for inds in [is_trn, is_holdout, is_val]:
            values = image_order[inds]
            image_order[inds] = values[np.random.permutation(len(values))]      

    return voxel_data, image_order, val_inds, holdout_inds, session_inds
   

def resize_image_tensor(x, newsize):
        tt = x.transpose((0,2,3,1))
        r  = np.ndarray(shape=x.shape[:1]+newsize+(x.shape[1],), dtype=tt.dtype) 
        for i,t in enumerate(tt):
            r[i] = np.asarray(PIL.Image.fromarray(t).resize(newsize, resample=PIL.Image.BILINEAR))
        return r.transpose((0,3,1,2))   

def get_subject_specific_images(nsd_root, path_to_save, npix=227, debug=False):

    """ 
    Load the big array of NSD images for all subjects.
    Downsample to a desired size, and select just those viewed by a given subject.
    Save a smaller array for each subject, at specified path.
    """
    
    stim_file_original = os.path.join(nsd_root,"nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5")
    exp_design_file = os.path.join(nsd_root,"nsddata/experiments/nsd/nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    subject_idx  = exp_design['subjectim']
    if debug:
        subject_idx = subject_idx[0:1]
        
    print ("Loading full block of images...")
    image_data_set = h5py.File(stim_file_original, 'r')
    print(image_data_set.keys())
    image_data = np.copy(image_data_set['imgBrick'])
    image_data_set.close()
    print(image_data.shape)

    for k,s_idx in enumerate(subject_idx):
        fn2save = os.path.join(path_to_save, 'S%d_stimuli_%d'%(k+1, npix))
        print('Will save to %s'%fn2save)       
        print('Resizing...')
        s_image_data = image_data[s_idx - 1]
        s_image_data = resize_image_tensor(s_image_data.transpose(0,3,1,2), newsize=(npix,npix))
        print(s_image_data.shape)        
        print('saving to %s'%fn2save)
        
        with h5py.File(fn2save + '.h5py', 'w') as hf:
            key='stimuli'
            val=s_image_data        
            hf.create_dataset(key,data=val)
            print ('saved %s in h5py file' %(key))


def get_subj_df(subject):
    """
    Get info about the 10,000 images that were shown to each subject.
    Note this is not the full ordered sequence of trials (which is 30,000 long)
    This is only the unique images 
    (matches what is in /user_data/mmhender/nsd_stimuli/stimuli/nsd/S1_stimuli....h5py)
    """
    exp_design_file = os.path.join(nsd_root,"nsddata/experiments/nsd/nsd_expdesign.mat")
    exp_design = loadmat(exp_design_file)
    subject_idx  = exp_design['subjectim']
    
    nsd_meta_file = os.path.join(nsd_root, 'nsddata/experiments/nsd/nsd_stim_info_merged.pkl')
    with open(nsd_meta_file,'rb') as f:
        stim_info = pickle.load(f,encoding="latin1")
    
    ss=subject-1
    subject_df = stim_info.loc[subject_idx[ss,:]-1]

    return subject_df


def load_prf_mapping_pars(subject, voxel_mask=None):
    
    """
    Load parameters of pRF fits for each voxel, obtained during independent pRF mapping expt.
    Stimuli are sweeping bars w objects, see:
    https://natural-scenes-dataset.s3-us-east-2.amazonaws.com/nsddata/experiments/prf/prf_screencapture.mp4
    """
    
    if voxel_mask is None:
        voxel_mask = get_voxel_mask(subject)
        
    prf_path = os.path.join(default_paths.nsd_root, 'nsddata','ppdata','subj%02d'%subject,'func1pt8mm')

    angle = load_from_nii(os.path.join(prf_path, 'prf_angle.nii.gz')).flatten()[voxel_mask]
    eccen = load_from_nii(os.path.join(prf_path, 'prf_eccentricity.nii.gz')).flatten()[voxel_mask]
    size = load_from_nii(os.path.join(prf_path, 'prf_size.nii.gz')).flatten()[voxel_mask]
    exponent = load_from_nii(os.path.join(prf_path, 'prf_exponent.nii.gz')).flatten()[voxel_mask]
    gain = load_from_nii(os.path.join(prf_path, 'prf_gain.nii.gz')).flatten()[voxel_mask]
    rsq = load_from_nii(os.path.join(prf_path, 'prf_R2.nii.gz')).flatten()[voxel_mask]/100
            
    return angle, eccen, size, exponent, gain, rsq


def load_domain_tvals(subject, voxel_mask=None):

    """
    For one NSD subject, load the t-statistics for all domain contrasts 
    from independent localizer task (faces, places, etc)
    """
    
    if voxel_mask is None:
        voxel_mask = get_voxel_mask(subject)
        
    n_voxels = np.sum(voxel_mask)

    niftis_path = os.path.join(default_paths.nsd_root, \
                               'nsddata', 'ppdata','subj%02d'%subject, 'func1pt8mm')

    categ_list = ['places', 'faces', 'bodies', 'objects', 'characters']
    n_categ = len(categ_list)

    tvals_all = np.zeros((n_voxels, n_categ))

    for cc, categ in enumerate(categ_list):

        # load t-statistics for the domain contrast of interest
        tvals_filename = os.path.join(niftis_path, 'floc_%stval.nii.gz'%categ)
        tvals = load_from_nii(tvals_filename)
        tvals = tvals.reshape((1, -1), order='C')[0]

        # pull out same set of voxels all my analyses were done on
        tvals_masked = tvals[voxel_mask] 

        tvals_all[:,cc] = tvals_masked

    return tvals_all, categ_list


def get_image_ranks(subject, sessions=np.arange(0,40), debug=False):
    
    """
    For each voxel, rank images in order of average response 
    (averaged over duplicate trials) and save as a csv file.
    Each column in csv is a voxel, each row is a rank position.
    """

    if np.isscalar(sessions):
        sessions = [sessions]
    sessions = np.array(sessions)
    max_sess_each_subj = [40,40,32,30,40,32,40,30]
    if np.any((sessions+1)>max_sess_each_subj[subject-1]):
        # adjust the session list that was entered, if the subject is missing some sessions.
        # will alter the list for both images and voxel data.
        print('subject %d only has up to session %d, will load these sessions:'%(subject, max_sess_each_subj[subject-1]))
        sessions = sessions[(sessions+1)<=max_sess_each_subj[subject-1]]
        print(sessions+1)
        assert(len(sessions)>0)
    if debug:
        sessions = np.array([0])
        
    voxel_mask = get_voxel_mask(subject)
        
    voxel_data = load_betas(subject, sessions, voxel_mask=voxel_mask, \
                              zscore_betas_within_sess=True, \
                              volume_space=True)    
    image_order = get_master_image_order()
    session_inds = get_session_inds_full()

    inds2use = np.isin(session_inds, sessions)
    image_order = image_order[inds2use]

    n_trials = voxel_data.shape[0]
    n_voxels = voxel_data.shape[1]

    unique_ims = np.unique(image_order)
    n_unique_ims = len(unique_ims)
    avg_dat_each_image = np.zeros((n_unique_ims, n_voxels))
    for uu, im in enumerate(unique_ims):
        if debug and (uu>1):
            continue
        inds = image_order==im;
        avg_dat_each_image[uu,:] = np.mean(voxel_data[inds,:], axis=0)


    images_ranked_each_voxel = np.zeros((n_unique_ims, n_voxels))
    for vv in range(n_voxels):
        if debug and (vv>1):
            continue
        image_rank = np.flip(np.argsort(avg_dat_each_image[:,vv]))
        images_ranked = unique_ims[image_rank]
        images_ranked_each_voxel[:,vv] = images_ranked

    rank_df = pd.DataFrame(data=images_ranked_each_voxel.astype(int), \
                           columns=['voxel %d'%vv for vv in range(n_voxels)])

    fn2save = os.path.join(default_paths.stim_root, 'S%d_ranked_images.csv'%subject)
    print('Saving to %s'%fn2save)
    rank_df.to_csv(fn2save, header=True)
    
def load_image_data_partitions(subject):
    
    fn2load = "/data6/shubham/PC/data/results_preprocessed/Image_data_partitions.npy"
    print()
    print('loading train/holdout/val image list from %s'%fn2load)
    partitions = np.load(fn2load, allow_pickle=True).item()
    is_trn = partitions['is_trn'][:,subject-1]
    is_holdout = partitions['is_holdout'][:,subject-1]
    is_val = partitions['is_val'][:,subject-1]
    
    return is_trn, is_holdout, is_val

def make_image_data_partitions(pct_holdout=0.10):

    ''' 
        return 3 matrices is_train, is_val, is_holdout each of dimension 10k * 9, first of 8 columns represent 8 subjects
        (ignore last column), i, e 0th column represent 1st subj and so on, if is_trn[4253, 4] = 1 this means the image with 
        nsd ind = 4253 is for training purpose for subj = 3
    '''

    subjects=np.concatenate([np.arange(1,9),[999]], axis=0)
    n_subjects = len(subjects)
    # fixed random seeds for each subject, to make sure shuffling is repeatable
    rndseeds = [171301, 42102, 490304, 521005, 11407, 501610, 552211, 450013, 824387]
    
    n_images_total = 10000
    is_trn = np.zeros((n_images_total,n_subjects),dtype=bool)
    is_holdout = np.zeros((n_images_total,n_subjects),dtype=bool)
    is_val = np.zeros((n_images_total,n_subjects),dtype=bool)

    for si, ss in enumerate(subjects):

        if ss==999:
            val_image_inds = np.arange(0,10000)<1000
            trn_image_inds = np.arange(0,10000)>=1000            
        else:           
            subject_df = get_subj_df(ss)
            val_image_inds = subject_df['shared1000']
            trn_image_inds = ~subject_df['shared1000']

        n_images_val = np.sum(val_image_inds)
        n_images_notval = np.sum(trn_image_inds);
        n_images_holdout = int(np.ceil(n_images_notval*pct_holdout))
        n_images_trn = n_images_notval - n_images_holdout

        # of the full 9000 image training set, holding out a random chunk
        inds_notval = np.where(trn_image_inds)[0]
        np.random.seed(rndseeds[si])
        np.random.shuffle(inds_notval)
        inds_trn = inds_notval[0:n_images_trn]
        inds_holdout = inds_notval[n_images_trn:]
        assert(len(inds_holdout)==n_images_holdout)

        is_trn[inds_trn,si] = 1
        is_holdout[inds_holdout,si] = 1
        is_val[val_image_inds,si] = 1

    # fn2save = os.path.join(default_paths.stim_root, 'Image_data_partitions.npy')
    fn2save = "/data6/shubham/PC/data/results_preprocessed/Image_data_partitions.npy"
    np.save(fn2save, {'is_trn': is_trn, \
                      'is_holdout': is_holdout, \
                      'is_val': is_val, \
                      'rndseeds': rndseeds})
    
    
def discretize_mappingtask_prfs(which_prf_grid=5):
    
    """
    Converting pRF definitions from the pRF mapping task (which are continous)
    into the closest parameters from a grid of pRFs
    Can be used for fitting models
    
    """

    prf_grid = initialize_fitting.get_prf_models(which_prf_grid).round(3)
    grid_x_deg, grid_y_deg = prf_grid[:,0]*8.4, prf_grid[:,1]*8.4
    grid_size_deg = prf_grid[:,2]*8.4

    subjects = np.arange(1,9)
    for si,ss in enumerate(subjects):

        voxel_mask = get_voxel_mask(subject=ss)
        n_vox = np.sum(voxel_mask)

        a,e,s, exp,gain,rsq = load_prf_mapping_pars(subject=ss, voxel_mask = voxel_mask)
        x_mapping, y_mapping = prf_utils.pol_to_cart(a,e)
        x_mapping = np.minimum(np.maximum(x_mapping, -7), 7)
        y_mapping = np.minimum(np.maximum(y_mapping, -7), 7)
        s_mapping = np.minimum(s, 8.4)
        
        print('there are %d nans in x_mapping'%np.sum(np.isnan(x_mapping)))
        print('there are %d nans in y_mapping'%np.sum(np.isnan(y_mapping)))
        print('there are %d nans in s_mapping'%np.sum(np.isnan(s_mapping)))
        x_mapping[np.isnan(x_mapping)] = 0
        y_mapping[np.isnan(y_mapping)] = 0

        prf_grid_inds = np.zeros((n_vox,1),dtype=int)

        for vv in range(n_vox):

            # first find the [x,y] coordinate closest to this pRF center (in my grid)
            distances_xy = np.sqrt((x_mapping[vv]-grid_x_deg)**2 + (y_mapping[vv]-grid_y_deg)**2)

            # should be multiple possible values here, for the different sizes
            closest_xy_inds = np.where(distances_xy==np.min(distances_xy))[0]

            # then find which size is closest to mapping task estimate
            distances_size = np.abs(s_mapping[vv] - grid_size_deg[closest_xy_inds])

            closest_ind = closest_xy_inds[np.argmin(distances_size)]

            prf_grid_inds[vv] = closest_ind.astype(int)

        save_prfs_folder = os.path.join(default_paths.save_fits_path, 'S%02d'%ss, 'mapping_task_prfs_grid%d'%which_prf_grid)

        if not os.path.exists(save_prfs_folder):
            os.makedirs(save_prfs_folder)

        save_filename = os.path.join(save_prfs_folder, 'prfs.npy')
        print('saving to %s'%save_filename)
        np.save(save_filename, {'voxel_mask': voxel_mask,  \
                                'prf_grid_inds': prf_grid_inds, \
                                'prf_grid_pars': prf_grid})
        
def coco_crop(img,cropbox_in):
    if type(cropbox_in) is str:
        cropbox = eval(cropbox_in)
    else:
        cropbox = cropbox_in
    top = int(img.shape[0]*cropbox[0])
    bottom = int(img.shape[0]*(1-cropbox[1]))
    left = int(img.shape[1]*cropbox[2])
    right = int(img.shape[1]*(1-cropbox[3]))
    return img[top:bottom, left:right]

def get_cocopath(coco_id_): #This is the modified code, modified by me
    coco_path = "/data6/shubham/PC/data/BrinDiVE_fromPrince/coco2017"
    if not isinstance(coco_id_, int):
        mycoco_id = str(int(coco_id_))
    else:
        mycoco_id = str(coco_id_)
    k = mycoco_id.zfill(12)+".jpg"
    f1 = os.path.join(coco_path, "train2017", k) 
    f2 = os.path.join(coco_path, "test2017", k) 
    f3 = os.path.join(coco_path, "val2017", k)
     
    if os.path.isfile(f1): return f1
        

    if os.path.isfile(f2): return f2
        

    if os.path.isfile(f3): return f3
        
    return None


if __name__ == "__main__":

    for subj in [1, 2, 5, 7]:  # only subjects 1, 2, 5, 7 are used 
        print()
        print("Starting subject ", subj)
        outs = get_data_splits(subj, sessions=list(range(40)), average_image_reps=True)  

        np.save("/data6/shubham/PC/data/results_preprocessed/NSD_zscored/subj_{}".format(subj), outs[0])
        np.save("/data6/shubham/PC/data/results_preprocessed/NSD_zscored/subj_{}_order".format(subj), outs[1])

        del outs
        gc.collect()

    try:
        with open("/data6/shubham/PC/data/BrinDiVE_fromPrince/nsd_data/nsddata/experiments/nsd/nsd_stim_info_merged.pkl", "rb") as f:
            data = pkl.load(f, encoding='latin1')  # Specify encoding to handle different character sets
    except UnicodeDecodeError as e:
        print(f"Error loading pickle file: {e}")
        raise

    df = pd.DataFrame(data)
    df.to_csv(r'/data6/shubham/PC/data/results_preprocessed/nsd_stim_info_merged.csv', index=False)
    csv_obj = pd.read_csv("/data6/shubham/PC/data/results_preprocessed/nsd_stim_info_merged.csv")

    all_coco = list(csv_obj["cocoId"])

    # file = h5py.File("/scratch/image_data.h5py", 'w')
    file = h5py.File("/data6/shubham/PC/data/results_preprocessed/image_data.h5py", 'w')   # this file is going to contain all 73k images shown to all subj
    zz = 0                                                                            # the name each image has its coco ID
    for i in all_coco:
        zz += 1
        if zz %100 == 0:
            print(zz)
        current_box = list(csv_obj[csv_obj["cocoId"]==int(i)]["cropBox"])[0]
        example_img = imageio.v2.imread(get_cocopath(i))
        cropped_image = np.array(coco_crop(example_img, current_box))

        cropped_image = Image.fromarray(cropped_image)
        output = np.array(cropped_image.resize((256, 256), resample=Image.Resampling.BICUBIC))
        file.create_dataset(str(i).zfill(12), data=output[:])

    file.close()











    


