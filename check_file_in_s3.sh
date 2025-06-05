#!/bin/bash
 
# File to check
file_to_check="bodies_S1_t2.npy"

 
# Root directory of the S3 bucket
bucket="s3://natural-scenes-dataset"
 
# Get all subdirectories
directories=$(aws s3 ls --no-sign-request $bucket/ | grep PRE | awk '{print $2}')
 
# Loop through directories and check for the file
for dir in $directories; do
    echo "Checking in directory: $dir"
    # Use --include to filter the grep to find the file directly
    result=$(aws s3 ls --no-sign-request "${bucket}/${dir}" --recursive | grep "$file_to_check")
    if [ -n "$result" ]; then
        echo "File $file_to_check found in $dir"
        echo $result
        break
    else
        echo "File $file_to_check not found in $dir"
    fi
done
 