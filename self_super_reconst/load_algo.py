import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

device = 'cuda' #@param ['cpu', 'cuda'] {allow-input: true}
device = torch.device(device)



class argObj:
    def __init__(self, data_dir, parent_submission_dir, subj):
        
        self.subj = format(subj, '02')
        self.data_dir = os.path.join(data_dir, 'subj'+self.subj)
        self.parent_submission_dir = parent_submission_dir
        self.subject_submission_dir = os.path.join(self.parent_submission_dir,
            'subj'+self.subj)

        # Create the submission directory if not existing
        if not os.path.isdir(self.subject_submission_dir):
            os.makedirs(self.subject_submission_dir)



class ImageDataset(Dataset):
    def __init__(self, imgs_paths, fmri, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.fmri = np.array(fmri)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        fmri_img = np.float32(self.fmri[idx])
        # rh_fmri_img = self.rh_fmri[idx]     # check about what to do with rh_fmri
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(device)
        return img, fmri_img
    

def get_NSD_datasets(sub, roi):

    data_dir = "/hdd_home/shank/algonauts/algonauts_2023_data/"
    parent_submission_dir = 'algonauts_2023_challenge_submission'


    subj = sub #@param ["1", "2", "3", "4", "5", "6", "7", "8"] {type:"raw", allow-input: true}


    args = argObj(data_dir, parent_submission_dir, subj)



    ### LOAD FMRI DATA

    fmri_dir = os.path.join(args.data_dir, 'masked_new')
    print(fmri_dir)
    # fmri_dir = os.path.join(args.data_dir, 'masked_new_temp')
    print('(Training stimulus images × {} )'.format(roi))
    fmri = np.load(os.path.join(fmri_dir, '{}_fmri.npy'.format(roi)))
    # rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))

    print('Training fMRI data shape:')
    print(fmri.shape)
    # print('(Training stimulus images × LH vertices)')

    # print('\nRH training fMRI data shape:')
    # print(rh_fmri.shape)
    # print('(Training stimulus images × RH vertices)')


    ### LOAD STIMULUS IMAGES


    train_img_dir  = os.path.join(args.data_dir, 'training_split', 'training_images')
    # test_img_dir  = os.path.join(args.data_dir, 'test_split', 'test_images')

    # Create lists will all training and test image file names, sorted
    train_img_list = os.listdir(train_img_dir)
    train_img_list.sort()
    # test_img_list = os.listdir(test_img_dir)
    # test_img_list.sort()
    print('Training images: ' + str(len(train_img_list)))
    # print('Test images: ' + str(len(test_img_list)))


    rand_seed = 5 #@param
    np.random.seed(rand_seed)

    # Calculate how many stimulus images correspond to 90% of the training data
    num_train = int(np.round(len(train_img_list) / 100 * 90))
    # Shuffle all training stimulus images
    idxs = np.arange(len(train_img_list))
    np.random.shuffle(idxs)
    # Assign 90% of the shuffled stimulus images to the training partition,
    # and 10% to the test partition
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    # No need to shuffle or split the test stimulus images
    # idxs_test = np.arange(len(test_img_list))

    print('Training stimulus images: ' + format(len(idxs_train)))
    print('\nValidation stimulus images: ' + format(len(idxs_val)))
    # print('\nTest stimulus images: ' + format(len(idxs_test)))



    # img_xfm_basic = transforms.Compose([
    #     transforms.Resize(size=(112,112), interpolation=Image.BILINEAR),
    #     transforms.CenterCrop((112,112)),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])

    img_xfm_basic = transforms.Compose([
        transforms.Resize(size=(256,256), interpolation=Image.BILINEAR),
        transforms.CenterCrop((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    img_xfm_train = transforms.Compose([
        transforms.Resize(size=(112,112), interpolation=Image.BILINEAR),
        transforms.RandomCrop(size=(112,112), padding=int(3 / 100 * 112), padding_mode='edge'),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    transform = transforms.Compose([
        transforms.Resize((224,224)), # resize the images to 224x24 pixels
        transforms.ToTensor(), # convert the images to a PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
    ])

    
    # batch_size = 1 #@param

    # batch_size = 250 #@param

    # batch_size = 180 #@param

    batch_size = 120 #@param

    # Get the paths of all image files
    train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
    # test_imgs_paths = sorted(list(Path(test_img_dir).iterdir()))

    # The DataLoaders contain the ImageDataset class
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, fmri, idxs_train, img_xfm_basic), 
        batch_size=batch_size
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, fmri, idxs_val, img_xfm_basic), 
        batch_size=batch_size
    )
    # test_imgs_dataloader = DataLoader(
    #     ImageDataset(test_imgs_paths, fmri, idxs_test, img_xfm_basic), 
    #     batch_size=batch_size
    # )


    return train_imgs_dataloader, val_imgs_dataloader#, test_imgs_dataloader
