import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import socket
from contextlib import closing
import numpy as np
import random
from time import time
import functools

from tqdm import tqdm

from encoder_dataloader_update import neural_loader
import encoder_model_vit_update
from encoder_options import Options

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def worker_init_fn(worker_id, myrank_info):
    np.random.seed(worker_id + myrank_info*100)

def shuffle_shift(input_image, extent=4):
    offset_x = random.randint(-extent, extent)
    offset_y = random.randint(-extent, extent)
    orig_shape = input_image.shape
    temp = input_image[:,:, max(0,offset_x):min(orig_shape[2], orig_shape[2]+offset_x), max(0,offset_y):min(orig_shape[3], orig_shape[3]+offset_y)]
    return torch.nn.functional.pad(temp, (max(0, -offset_y),max(0,offset_y), max(0, -offset_x), max(0,offset_x)), mode='replicate')

def train_net(rank, world_size, freeport, other_args):
    # Set up distributed training environment
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = freeport
    output_device = rank + other_args.gpu_id
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(output_device)
    torch.backends.cudnn.benchmark = True

    # Prepare dataset and dataloader
    dataset = neural_loader(other_args)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    ranked_worker_init = functools.partial(worker_init_fn, myrank_info=rank)

    neural_dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=other_args.batch_size//world_size, 
        shuffle=False, 
        num_workers=4, 
        worker_init_fn=ranked_worker_init, 
        persistent_workers=True, 
        sampler=train_sampler,
        drop_last=False
    )

    # Initialize model components
    feature_extractor = encoder_model_vit_update.feature_extractor_vit([3,5])
    feature_extractor.to(output_device)
    feature_extractor.eval()

    projector = encoder_model_vit_update.downproject_CLIP_split_linear_higher(num_higher_output=dataset.sizes)
    projector.train()
    projector.to(output_device)

    # Handle resuming training from checkpoint
    start_epoch = 1
    load_opt = 0
    loaded_weights = False
    if other_args.resume:
        if not os.path.isdir(other_args.exp_dir):
            print("Missing save dir, exiting")
            dist.barrier()
            dist.destroy_process_group()
            return 1

        current_files = sorted(os.listdir(other_args.exp_dir))
        if current_files:
            latest = current_files[-1]
            start_epoch = int(latest.split(".")[0]) + 1
            
            if start_epoch >= (other_args.epochs+1):
                dist.barrier()
                dist.destroy_process_group()
                return 1
            
            map_location = 'cpu'
            weight_loc = os.path.join(other_args.exp_dir, latest)
            weights = torch.load(weight_loc, map_location=map_location)
            
            projector.load_state_dict(weights["network"])
            loaded_weights = True
            if "opt" in weights:
                load_opt = 1

        if not loaded_weights:
            print("Resume indicated, but no weights found!")
            dist.barrier()
            dist.destroy_process_group()
            exit()

    # Distributed Data Parallel setup
    ddp_projector = DDP(projector, find_unused_parameters=False, device_ids=[output_device], gradient_as_bucket_view=True)

    # Optimizer setup
    criterion = torch.nn.MSELoss()
    decay = [m for name, m in ddp_projector.named_parameters() if "higher" in name]
    no_decay = [m for name, m in ddp_projector.named_parameters() if "higher" not in name]

    optimizer = torch.optim.AdamW([
        {'params': decay, 'lr': other_args.lr_init, 'weight_decay': 2e-2},
        {'params': no_decay, 'lr': other_args.lr_init, 'weight_decay': 1.5e-2}
    ], lr=other_args.lr_init, weight_decay=1.5e-2)

    if load_opt:
        optimizer.load_state_dict(weights["opt"])

    # Training loop with progress bars
    for epoch in range(start_epoch, other_args.epochs+1):
        # Epoch-level progress bar (only for rank 0)
        if rank == 0:
            epoch_pbar = tqdm(total=other_args.epochs, desc=f"Training Epochs", position=0)
            epoch_pbar.update(epoch)

        # Learning rate decay
        decay_rate = other_args.lr_decay
        new_lrate = other_args.lr_init * (decay_rate ** (epoch / other_args.epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        total_losses = 0
        cur_iter = 0
        train_sampler.set_epoch(epoch)

        # Batch-level progress bar (only for rank 0)
        if rank == 0:
            batch_pbar = tqdm(neural_dataloader, desc=f"Epoch {epoch}", position=1, leave=False)
        else:
            batch_pbar = neural_dataloader

        for data_stuff in batch_pbar:
            neural_data = data_stuff["neural_data"].to(output_device, non_blocking=True)
            image_data = data_stuff["image_data"][:,0].to(output_device, non_blocking=True)
            subj_order = data_stuff["subject_id"].reshape(-1).tolist()
            
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda"):
                with torch.no_grad():
                    features = feature_extractor(shuffle_shift(image_data)+torch.randn_like(image_data)*0.05)
                
                predicted = ddp_projector(features[0][0].float(), features[0][1].float(), features[1].float(), subj_order)
                loss = criterion(predicted, neural_data)

            if rank==0:
                total_losses += loss.detach()
                cur_iter += 1
                # Update batch progress bar
                if hasattr(batch_pbar, 'set_postfix'):
                    batch_pbar.set_postfix({'Loss': loss.item()})

            loss.backward()
            optimizer.step()

        # Logging and checkpointing for rank 0
        if rank == 0:
            # Close batch progress bar for this epoch
            if hasattr(batch_pbar, 'close'):
                batch_pbar.close()

            avg_loss = total_losses.item() / cur_iter
            print(f"{other_args.exp_name}: Epoch {epoch}, Avg Loss: {avg_loss:.4f}, LR: {new_lrate}")

            # Save checkpoint periodically
            if epoch % 20 == 0 or epoch == 1 or epoch > (other_args.epochs-3):
                save_name = str(epoch).zfill(5)+".chkpt"
                save_dict = {"network": ddp_projector.module.state_dict()}
                torch.save(save_dict, os.path.join(other_args.exp_dir, save_name))
                
                # Progress bar for checkpointing
                tqdm.write(f"Checkpoint saved: {save_name}")

            # Update epoch progress bar
            epoch_pbar.update(1)

        dist.barrier()

    # Cleanup
    if rank == 0:
        epoch_pbar.close()

    dist.barrier()
    dist.destroy_process_group()
    return 1

if __name__ == '__main__':
    cur_args = Options().parse()
    cur_args.exp_name = "subject_{}_neurips_split_VIT_last_fully_linear"

    # Process subject IDs
    if len(cur_args.subject_id[0]) > 1:
        cur_args.subject_id = sorted([str(int(sbjid)) for sbjid in cur_args.subject_id[0].split(",")])
    
    # Prepare experiment name and directory
    exp_name_filled = cur_args.exp_name.format("-".join(cur_args.subject_id))
    cur_args.exp_name = exp_name_filled

    # Create save directory if not exists
    os.makedirs(os.path.join(cur_args.save_loc, exp_name_filled, 
                cur_args.functional1 + cur_args.functional2, 
                cur_args.region1 + cur_args.region2), 
                exist_ok=True)

    # Set experiment directory
    cur_args.exp_dir = os.path.join(cur_args.save_loc, exp_name_filled, 
                                     cur_args.functional1 + cur_args.functional2, 
                                     cur_args.region1 + cur_args.region2)

    # Run training
    world_size = cur_args.gpus
    myport = str(find_free_port())
    train_net(0, world_size, myport, cur_args)