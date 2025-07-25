import h5py
import numpy as np
import nibabel as nib
def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()
import os
import pickle
from scipy.io import loadmat
from tqdm import tqdm 

# nsd_root = '/lab_data/tarrlab/common/datasets/NSD'

nsd_root ='/data6/shubham/PC/data/BrinDiVE_fromPrince/nsd_data'

# for s in [1,2,3,4,5,6,7,8]:
for s in [1, 2, 5, 7]:  #we have only 4 subjects
    # selected = []

    # Get the voxels that belong to cortex. Basically anywhere the mask is 0 or greater.
    # for roi_strings in ["prf-visualrois.nii.gz","floc-bodies.nii.gz", "floc-faces.nii.gz", "floc-places.nii.gz", "floc-words.nii.gz", "HCP_MMP1.nii.gz", "Kastner2015.nii.gz", "nsdgeneral.nii.gz"]:
    #     full_path = "{}/nsddata/ppdata/subj0{}/func1pt8mm/roi/{}".format(nsd_root, s, roi_strings)
    #     all_roi = load_from_nii(full_path)
    #     selected.append(all_roi>=-0.5)
    # all_good = np.logical_or.reduce(selected)
    # print(np.sum(all_good))
    # print(s, " SUBJECT")

    print(f"starting with subject {s}")
    
    nsd_meta_file = "/data6/home2/spshubham/prashant/data/results_preprocessed/nsd_stim_info_merged.pkl"
    with open(nsd_meta_file, 'rb') as f:
        stim_info = pickle.load(f, encoding="latin1")
    exp_design_file = "/data6/home2/spshubham/prashant/data/results_preprocessed/nsd_expdesign.mat"
    exp_design = loadmat(exp_design_file)    
    subject_idx_MATRIX = exp_design['subjectim']    # 8 * 10k matrix

    subject_df = stim_info.loc[subject_idx_MATRIX[s - 1, :] - 1]
    COCO_ids_unorder = np.array(subject_df["cocoId"].tolist()).astype(np.int64)

    # Data loaded in here
    
    # Change the path to where you set it in Z_score.ipynb
    # full_data = np.load("/lab_data/tarrlab/afluo/NSD_zscored/subj_{}.npy".format(s)) #
    # order = np.load("/lab_data/tarrlab/afluo/NSD_zscored/subj_{}_order.npy".format(s))

    full_data = np.load("/data6/home2/spshubham/prashant/data/results_preprocessed/NSD_zscored/subj_{}.npy".format(s)) #
    order = np.load("/data6/home2/spshubham/prashant/data/results_preprocessed/NSD_zscored/subj_{}_order.npy".format(s))

    COCO_ids_ordered = COCO_ids_unorder[order]
    COCO_ids_ordered = COCO_ids_ordered.tolist()

    # Change the location to where you want to save it
    # file = h5py.File("/lab_data/tarrlab/afluo/NSD_zscored/cortex_subj_{}.npy".format(s), 'w')
    file = h5py.File("/data6/home2/spshubham/prashant/data/results_preprocessed/cortex/cortex_subj_{}.npy".format(s), 'w')

    # The mask is an object that goes from 3D full grid to just the cortical voxels
    # file.create_dataset("mask", data=all_good)

    for index in tqdm(range(len(COCO_ids_ordered)), desc=f"Processing Subject {s}", unit="COCO ID"):

        # However remember that Maggie's script gives us data in a flat 1D array, so you want to flatten the mask to use it.
        # dset = file.create_dataset(str(COCO_ids_ordered[index]), data=full_data[index][all_good.flatten()].astype(np.single))
        dset = file.create_dataset(str(COCO_ids_ordered[index]), data=full_data[index].astype(np.single))
    file.close()