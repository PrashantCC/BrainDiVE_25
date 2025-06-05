#!/bin/bash

# Initialize subject and GPU
subject_id=5
gpu=1

# Script name
script_name="run_brain_guidance_category_gaziv_parser.py"
  # Replace with your actual Python script name

# Sequential commands for each region
# python $script_name --subject_id $subject_id --region V1v --gpu $gpu
# python $script_name --subject_id $subject_id --region V1d --gpu $gpu
# python $script_name --subject_id $subject_id --region V2v --gpu $gpu
# python $script_name --subject_id $subject_id --region V2d --gpu $gpu
# python $script_name --subject_id $subject_id --region V3v --gpu $gpu
# python $script_name --subject_id $subject_id --region V3d --gpu $gpu
# python $script_name --subject_id $subject_id --region hV4 --gpu $gpu
# python $script_name --subject_id $subject_id --region EBA --gpu $gpu
# python $script_name --subject_id $subject_id --region FBA-2 --gpu $gpu
# python $script_name --subject_id $subject_id --region OFA --gpu $gpu
# python $script_name --subject_id $subject_id --region FFA-1 --gpu $gpu
# python $script_name --subject_id $subject_id --region FFA-2 --gpu $gpu
# python $script_name --subject_id $subject_id --region aTL-faces --gpu $gpu
# python $script_name --subject_id $subject_id --region OPA --gpu $gpu
# python $script_name --subject_id $subject_id --region PPA --gpu $gpu
python $script_name --subject_id $subject_id --region RSC --gpu $gpu
python $script_name --subject_id $subject_id --region OWFA --gpu $gpu
python $script_name --subject_id $subject_id --region VWFA-1 --gpu $gpu
python $script_name --subject_id $subject_id --region VWFA-2 --gpu $gpu
python $script_name --subject_id $subject_id --region mfs-words --gpu $gpu
python $script_name --subject_id $subject_id --region mTL-words --gpu $gpu
