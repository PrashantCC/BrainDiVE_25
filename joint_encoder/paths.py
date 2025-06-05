from pathlib import Path

root = Path("/data6/shubham/PC/")

image_path1 = Path("/data/image/image_data.h5py")
neural_path1 = Path("/data/cortex/cortex_subj_{}.npy")
mask_path1 = Path("/data/masks/subj0{}/{}.nii.gz")
sav_loc1 = Path("/data/results2/checkpoints")
weights_path1 = Path("/data/results2/checkpoints/subject_{}_neurips_split_VIT_last_fully_linear/{}/{}/00080.chkpt")

# gen_img_path1 = Path("/data/results1/Gaziv_gen_images/sub_{}/{}/{}/")
gen_img_path1 = Path("/data/results2/BrainDiVE_gen_images/sub_{}/{}/{}/")

image_path = root / image_path1.relative_to('/')
neural_path = str(root / neural_path1.relative_to('/'))
mask_path = str(root / mask_path1.relative_to('/'))
sav_loc = str(root / sav_loc1.relative_to('/'))
weights_path = str(root / weights_path1.relative_to('/'))
gen_img_path = str(root / gen_img_path1.relative_to('/'))

# print(image_path)
# print(neural_path.format("1"))
# print(mask_path.format("1", "prf-visualrois"))
# print(sav_loc)


