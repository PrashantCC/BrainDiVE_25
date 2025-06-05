gpu=0  # GPU ID
export CUDA_VISIBLE_DEVICES=$gpu
sbj_num=1
tensorboard_log_dir=/data6/shubham/PC/data/results_jointencoder_master_gaziv/tensorboard/sub1
 python self_super_reconst/train_encoder.py \
--exp_prefix sub${sbj_num}_rgb_only \
--separable 1 --n_epochs 50 --learning_rate 1e-3 --cos_loss 0.3 --random_crop_pad_percent 3 --scheduler 10 --gamma 0.2 \
--fc_gl 1 --fc_mom2 10 --l1_convs 1e-4 --is_rgbd 0 --allow_bbn_detach 1 --train_bbn 0 --norm_within_img 1 --may_save 1 \
--sbj_num $sbj_num --tensorboard_log_dir $tensorboard_log_dir --gpu $gpu \
--neural_activity_path "/data6/shubham/PC/data/cortex1/cortex_subj_{}.npy" \
--image_path "/data6/shubham/PC/data/image1/image_data.h5py"