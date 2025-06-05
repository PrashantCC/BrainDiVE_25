import os
import random
import pickle
import numpy as np
import torch
import h5py
import gc
import nibabel as nib
from skimage.transform import resize

# Set seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Constants for image normalization
OPENAI_CLIP_MEAN = np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.single)[:, None, None]
OPENAI_CLIP_STD = np.array((0.26862954, 0.26130258, 0.27577711), dtype=np.single)[:, None, None]

def load_from_nii(nii_file):
    return nib.load(nii_file).get_fdata()

def normalize_image(input_ndarray):
    image_resized = resize(input_ndarray, (224, 224), preserve_range=True)
    scaled_image = image_resized.astype(np.single).transpose((2, 0, 1)) * random.uniform(0.95, 1.05) / 255.0
    return (scaled_image - OPENAI_CLIP_MEAN) / OPENAI_CLIP_STD

def normalize_image_deterministic(input_ndarray):
    image_resized = resize(input_ndarray, (224, 224), preserve_range=True)
    scaled_image = image_resized.astype(np.single).transpose((2, 0, 1)) / 255.0
    return (scaled_image - OPENAI_CLIP_MEAN) / OPENAI_CLIP_STD

class neural_loader(torch.utils.data.Dataset):
    def __init__(self, arg_stuff):
        # self.subject_id = [arg_stuff.subject_id] if isinstance(arg_stuff.subject_id, int) else arg_stuff.subject_id
        self.subject_id = [int(sid) for sid in arg_stuff.subject_id] if isinstance(arg_stuff.subject_id, list) else [int(arg_stuff.subject_id)]

        self.neural_activity_path = arg_stuff.neural_activity_path
        self.image_path = arg_stuff.image_path
        self.functional = [
            {"name": arg_stuff.functional1, "region_idx": arg_stuff.i1},
            {"name": arg_stuff.functional2, "region_idx": arg_stuff.i2},
        ]
        self.transform = normalize_image

        # Metadata dictionaries
        self.all_keys = {}
        self.num_stimulus = {}
        self.neural_sizes = {}
        self.masks = {}
        self.sizes = {}

        print("Caching image_ids, this may take a while...")
        self.image_data = None

        # Load or compute all_keys
        self._load_or_cache_all_keys()

        # Identify common testing set across subjects
        self.testing_set = sorted(set.intersection(*[set(k) for k in self.all_keys.values()]))

        for subject in self.subject_id:
            self._process_subject_data(str(subject))

    def _load_or_cache_all_keys(self):
        if not os.path.exists("all_keys.pkl"):
            all_keys = {}
            for subject in [1, 2, 5, 7]:
                with h5py.File(self.neural_activity_path.format(subject), 'r') as f:
                    all_keys[str(subject)] = [k for k in f.keys() if k != "mask"]
            with open("all_keys.pkl", "wb") as f:
                pickle.dump(all_keys, f)
        with open("all_keys.pkl", "rb") as f:
            self.all_keys = pickle.load(f)

    def _process_subject_data(self, str_subject):
        with h5py.File(self.neural_activity_path.format(str_subject), 'r') as neural_data:
            self.all_keys[str_subject] = [k for k in self.all_keys[str_subject] if k not in self.testing_set]
            self.num_stimulus[str_subject] = len(self.all_keys[str_subject])
            self.neural_sizes[str_subject] = len(neural_data[self.all_keys[str_subject][0]][:])

        # Load functional masks and compute sizes
        self.masks[str_subject] = []
        for func in self.functional:
            
            functional_path =  f"/data6/shubham/PC/data/masks/subj0{str_subject}/{func['name']}.nii.gz"
            mask = (load_from_nii(functional_path) == func['region_idx']).flatten()
            self.masks[str_subject].append(mask)
        self.sizes[str_subject] = sum(mask.sum() for mask in self.masks[str_subject])

    def __len__(self):
        return max(self.num_stimulus.values())

    def __getitem__(self, idx):
        all_images, all_neural = [], []
        for subject_idx in self.subject_id:
            subject_key = str(subject_idx)
            neural_data = self._load_neural_data(subject_key)
            image_data = self._load_image_data()

            curidx = idx if idx < self.num_stimulus[subject_key] else random.randint(0, self.num_stimulus[subject_key] - 1)
            neural_key = self.all_keys[subject_key][curidx]

            # Combine neural data for all masks
            brain_act = np.concatenate(
                [neural_data[neural_key][:][mask] for mask in self.masks[subject_key]]
            )

            selected_image = self.transform(image_data[str(neural_key).zfill(12)][:])
            all_images.append(selected_image)
            all_neural.append(brain_act)

        all_neural = np.concatenate(all_neural)
        all_images = np.array(all_images, dtype=np.float32)
    
        return {
            "subject_id": torch.tensor(self.subject_id, dtype=torch.int32),
            "neural_data": torch.tensor(all_neural),
            "image_data": torch.tensor(all_images),
        }
    

    def _load_neural_data(self, subject_key):
        attr_name = f"subj_{subject_key}_neural_data"
        if not hasattr(self, attr_name) or getattr(self, attr_name) is None:
            setattr(self, attr_name, h5py.File(self.neural_activity_path.format(subject_key), 'r'))
        return getattr(self, attr_name)

    def _load_image_data(self):
        if self.image_data is None:
            self.image_data = h5py.File(self.image_path, 'r')
        return self.image_data
