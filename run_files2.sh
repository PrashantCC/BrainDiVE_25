#!/bin/bash

for sub in 1 2 5 7; do
    # Fix j1 = 1.0, vary j2 from -1.0 to 1.0
    for j2 in $(seq -1.0 0.2 1.0); do 
        echo "Running process with j1=1.0 and j2=$j2"
        python run_brain_guidance_category_gaziv_parser_combn.py \
            --subject_id $sub \
            --region1 V1v \
            --i1 2 \
            --j1 1.0 \
            --region2 FFA-1 \
            --i2 2 \
            --j2 $j2 \
            --gpu 1
    done

    # Fix j2 = 1.0, vary j1 from -1.0 to 1.0
    for j1 in $(seq -1.0 0.2 1.0); do 
        echo "Running process with j1=$j1 and j2=1.0"
        python run_brain_guidance_category_gaziv_parser_combn.py \
            --subject_id $sub \
            --region1 V1v \
            --i1 2 \
            --j1 $j1 \
            --region2 FFA-1 \
            --i2 2 \
            --j2 1.0 \
            --gpu 1
    done
done