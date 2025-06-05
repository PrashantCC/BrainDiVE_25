# Save this as run_encoder_dataloader.py
from encoder_options import Options
from encoder_dataloader_update import neural_loader

if __name__ == "__main__":
    opt = Options().parse()
    dataset = neural_loader(opt)
    print(f"Dataset length: {len(dataset)}")
    datapoint = dataset[0]
    # datapoint["subject_id"], datapoint["neural_data"], datapoint["image_data"]
    print(f"Subject_id: {datapoint["subject_id"]}")
    print(f"Sample neural shape: {datapoint["neural_data"].shape}")
    print(f"Sample neural shape: {datapoint["image_data"].shape}")
    print(f"early_sizes = {dataset.early_sizes}")
    
    print(f"higher_sizes = {dataset.higher_sizes}")
    print(sum(list(dataset.early_sizes.values())) + sum(list(dataset.higher_sizes.values())))