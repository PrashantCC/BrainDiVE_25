# # Setting paths of interest for this project
# # See /code/utils/default_paths.py for how these paths will be used to
# # look for files and folders.

# # Set path to your top-level directory
# # In here should be the subfolders:
# #     "nsd" which includes our pre-processed images and labels
# #     "modfit" (main project folder with all our code, also where this file lives)
# #     "features" (precomputed features extracted for each pRF)
# root = '/user_data/mmhender/'
# project_name = 'modfit'

# # if using a scratch directory local to the node i'm on, what is its path
# # (not used often)
# root_localnode = '/scratch/mmhender/'

# # Set path to the full NSD data repository (this is where the beta weights are stored)
# # http://naturalscenesdataset.org/
# nsd_path = '/lab_data/tarrlab/common/datasets/NSD'   

# # Path to the COCO API toolbox
# # https://github.com/cocodataset/cocoapi
# coco_api_path = '/user_data/mmhender/toolboxes/coco_annot'

# # Path to where the raw COCO images are stored
# # https://cocodataset.org/
# coco_ims_path = '/lab_data/tarrlab/common/datasets/COCO'


# # only need this path if using floc stimuli (categories)
# floc_image_root = '/lab_data/tarrlab/maggie/fLoc_stimuli/'
# # only need this path if using food imagea
# food_image_root = '/lab_data/tarrlab/maggie/food_images/'

# # path to trained models from starting blurry project
# startingblurry_root = '/home/ojinsi/trials/'

# retinaface_path = '/user_data/mmhender/toolboxes/RetinaFace-tf2/'




# My Update




# Setting paths of interest for this project

# Set path to your top-level directory
# root = '/media/internal_8T/prashant/BrainDiVE/'
root = "/data6/shubham/PC/data/BrinDiVE_fromPrince"
project_name = 'modfit'

# if using a scratch directory local to the node i'm on, what is its path
root_localnode = '/prashant/'

# Set path to the full NSD data repository
# nsd_path = '/media/internal_8T/prashant/BrainDiVE/nsd_data'   
nsd_path = "/data6/shubham/PC/data/BrinDiVE_fromPrince/nsd_data"

# Path to the COCO API toolbox
# coco_api_path = '/media/internal_8T/prashant/BrainDiVE/coco2017'  # Update this path as needed
coco_api_path = "/data6/shubham/PC/data/BrinDiVE_fromPrince/coco2017"

# Path to where the raw COCO images are stored
# coco_ims_path = "/media/internal_8T/prashant/BrainDiVE/coco2017"  # Update this path as needed
coco_ims_path = "/data6/shubham/PC/data/BrinDiVE_fromPrince/coco2017"


# only need this path if using floc stimuli (categories)
floc_image_root = '/path/to/your/floc-stimuli'  # Update this path as needed

# only need this path if using food images
food_image_root = '/path/to/your/food-images'  # Update this path as needed

# path to trained models from starting blurry project
startingblurry_root = '/path/to/your/starting-blurry-project'  # Update this path as needed

retinaface_path = '/path/to/your/retinaface-toolbox'  # Update this path as needed
