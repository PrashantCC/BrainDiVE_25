#!/bin/bash

# Loop through j1 values from 0.2 to 1.0 in increments of 0.2
for j1 in $(seq 0.2 0.2 1.0); do 
    echo "Running process with j1=$j1 and constant j2=1.0"
    python run_brain_guidance_category_braindive_parser.py \
        --subject_id 1 \
        --functional1 floc-faces \
        --region1 FFA-1 \
        --i1 2 \
        --j1 $j1 \
        --functional2 floc-places \
        --region2 PPA \
        --i2 2 \
        --j2 1.0 \
        --gpu 2
done
