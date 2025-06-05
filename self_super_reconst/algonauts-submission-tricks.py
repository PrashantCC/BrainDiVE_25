import numpy as np
import os
from tqdm import tqdm
from scipy import stats


# Z-scored
# gaziv_base_loc = "/hdd_home/achin/projs/ss_recon/SelfSuperReconst/self_super_reconst/gaziv_orig-z_scored_algonauts_2023_challenge_submission/"

# Without Z-score
gaziv_base_loc = "/hdd_home/achin/projs/ss_recon/SelfSuperReconst/self_super_reconst/algonauts_2023_challenge_submission/"

shank_base_loc = "/hdd_home/downloads/shank/algonauts_2023_challenge_submission/"
"/hdd_home/achin/projs/ss_recon/SelfSuperReconst/self_super_reconst/algonauts_2023_challenge_submission-2/subj01/lh_pred_test.npy"

# output_loc = "/hdd_home/achin/projs/ss_recon/SelfSuperReconst/self_super_reconst/tricks_algonauts_2023_challenge_submission/"

output_loc = "/hdd_home/achin/projs/ss_recon/SelfSuperReconst/self_super_reconst/basic_add_tricks_algonauts_2023_challenge_submission/"


for subj in tqdm(range(1, 9)):
    lh_gaziv_test = np.load(gaziv_base_loc + "subj%02d/"%subj + "lh_pred_test.npy")
    rh_gaziv_test = np.load(gaziv_base_loc + "subj%02d/"%subj + "rh_pred_test.npy")

    lh_shank_test = np.load(shank_base_loc + "subj%02d/"%subj + "lh_pred_test.npy")
    rh_shank_test = np.load(shank_base_loc + "subj%02d/"%subj + "rh_pred_test.npy")

    # Basic addition of predictions
    lh_pred_test = (10)*lh_gaziv_test + (1)*lh_shank_test
    rh_pred_test = (10)*rh_gaziv_test + (1)*rh_shank_test


    # import pdb; pdb.set_trace()

    # lh_shank_test = stats.zscore(lh_shank_test, axis=1)
    # rh_shank_test = stats.zscore(rh_shank_test, axis=1)

    # idx_lh = np.where(lh_gaziv_test == 0)
    # lh_pred_test = np.zeros_like(lh_gaziv_test)
    # lh_pred_test[idx_lh[0], idx_lh[1]] = lh_shank_test[idx_lh[0], idx_lh[1]]

    # idx_rh = np.where(rh_gaziv_test == 0)
    # rh_pred_test = np.zeros_like(rh_gaziv_test)
    # rh_pred_test[idx_rh[0], idx_rh[1]] = rh_shank_test[idx_rh[0], idx_rh[1]]


    # lh_pred_test += lh_gaziv_test + lh_shank_test
    # rh_pred_test += rh_gaziv_test + rh_shank_test

    ##############################################
    ## Z-score except

    # idx_lh = np.where(lh_gaziv_test == 0)
    # lh_pred_test = np.zeros_like(lh_gaziv_test)
    # lh_pred_test[idx_lh[0], idx_lh[1]] = lh_shank_test[idx_lh[0], idx_lh[1]]

    # idx_rh = np.where(rh_gaziv_test == 0)
    # rh_pred_test = np.zeros_like(rh_gaziv_test)
    # rh_pred_test[idx_rh[0], idx_rh[1]] = rh_shank_test[idx_rh[0], idx_rh[1]]

    # lh_shank_test = stats.zscore(lh_shank_test, axis=1)
    # rh_shank_test = stats.zscore(rh_shank_test, axis=1)

    # lh_pred_test += lh_gaziv_test + lh_shank_test
    # rh_pred_test += rh_gaziv_test + rh_shank_test

    ##############################################


    # lh_pred_test += lh_gaziv_test
    # rh_pred_test += rh_gaziv_test


    # lh_pred_test = (lh_gaziv_test + lh_shank_test)/2.0
    # rh_pred_test = (rh_gaziv_test + rh_shank_test)/2.0


    # idx_lh = np.where(lh_gaziv_test == 0)
    # # lh_pred_test = np.zeros_like(lh_gaziv_test)
    # lh_pred_test[idx_lh[0], idx_lh[1]] = 0.0

    # idx_rh = np.where(rh_gaziv_test == 0)
    # # rh_pred_test = np.zeros_like(rh_gaziv_test)
    # rh_pred_test[idx_rh[0], idx_rh[1]] = 0.0


    lh_pred_test = np.float32(lh_pred_test)
    rh_pred_test = np.float32(rh_pred_test)

    if(not os.path.exists(output_loc + "subj%02d/"%subj)):
        os.mkdir(output_loc + "subj%02d/"%subj)

    np.save(output_loc + "subj%02d/"%subj + "lh_pred_test.npy", lh_pred_test)
    np.save(output_loc + "subj%02d/"%subj + "rh_pred_test.npy", rh_pred_test)




