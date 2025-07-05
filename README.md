# 🧠 BrainDiVE – Joint Encoder Experiments

This repository contains all the experiments performed using **ViT-based** and **Gaziv-based** encoders under the BrainDiVE framework. Each Git branch corresponds to a distinct experimental configuration.

> 📂 **Code Path:**  
> `/data6/home2/spshubham/prashant/scripts/BrainDiVE/joint_encoder_master`

---

## 🔶 ViT-Based Encoder Experiments

### `vit_clip`
- **Description**: Baseline BrainDiVE encoder (same as the original BrainDiVE).
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_jointencoder_master_vit_clip`

---

### `vit_clip_algonauts`
- **Description**: Encoder trained on the **Algonauts dataset** using **full fMRI**.
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_jointencoder_master_vit_clip_algonauts`

---

### `vit_clip_mean`
- **Description**: Encoder trained on the **NSD dataset** using **mean fMRI**.
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_jointencoder_master_vit_clip_mean`

---

### `vit_clip_mean_general`
- **Description**: Encoder trained on **NSD mean fMRI** of subjects **1, 2, 5, and 7**.
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_jointencoder_master_vit_clip_mean_general`

---

### `vit_clip_mean_general_any3`
- **Description**: Encoder trained on **NSD mean fMRI** of subjects **1, 2, and 5**.
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_jointencoder_master_vit_clip_mean_general_125`

---

### `vit_dino`
- **Description**: Encoder use ViT trained on DINO task(distillatiuon with no labels).
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_jointencoder_master_VIT_DINO`
---

### `vit_imagenet`
- **Description**: Encoder uses ViT trained on classification task with Imagenet dataset
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_jointencoder_master_VIT_imagenet`
---

## 🔷 Gaziv-Based Encoder Experiments

### `main`
- **Description**: Original Gaziv encoder trained on the **Algonauts dataset** with **full fMRI**.
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results1`

---

### `comb_all_mean`
- **Description**: Gaziv encoder trained on **NSD** using **mean fMRI**.
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_gaziv_combn_all_mean`

---

### `comb_all_mean_gen`
- **Description**: Gaziv encoder trained on **NSD mean fMRI** of subjects **1, 2, 5, and 7**.
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_gaziv_combn_all_mean_gen`

---

### `only26`
- **Description**: Gaziv encoder using **only the 26th layer** of **VGGNet** for fMRI prediction.
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_jointencoder_master_gaziv/combn_26`

---

### `only3`
- **Description**: Gaziv encoder using **only the 3rd layer** of **VGGNet** for fMRI prediction.
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_jointencoder_master_gaziv/combn_3`

---

### `only3&26`
- **Description**: Gaziv encoder using a **combination of 3rd and 26th layers** of **VGGNet** for fMRI prediction.
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_jointencoder_master_gaziv/combn_3&26`

---

### `only_8&17`
- **Description**: Gaziv encoder using a **combination of 8th and 17th layers** of **VGGNet** for fMRI prediction.
- **Results Path**:  
  `/data6/home2/spshubham/prashant/data/results_jointencoder_master_gaziv/combn_8&17`

---

Explore each branch to dive into specific implementation and results. For any issues or questions, feel free to raise an issue in this repository.
