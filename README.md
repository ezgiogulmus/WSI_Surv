# MIL Models for Discrete-Time Survival Prediction

This repository contains an adaptation of the CLAM (Clustering-constrained Attention Multiple Instance Learning) and TransMIL (Transformer based Correlated Multiple Instance Learning) models for discrete-time survival prediction tasks using Whole-Slide Images.

## Installation

Tested on:
- Ubuntu 22.04
- Nvidia GeForce RTX 4090
- Python 3.10
- PyTorch 2.3

Clone the repository and go to the directory.

```bash
git clone https://github.com/ezgiogulmus/WSI_Surv.git
cd WSI_Surv
```

Install conda environment and required packages.

```bash
conda env create -n wsi_surv python=3.10 -y
conda activate wsi_surv
pip install --upgrade pip 
pip install -e .
```

## Usage

First, extract patch coordinates and patch-level features using the CLAM library available at [CLAM GitHub](https://github.com/Mahmoodlab/CLAM). Then, run the following command:

```bash
python main.py --split_dir name_of_the_split_folder --model_type clam_sb --feats_dir path/to/features_directory
```

- `model_type`: Options are `clam_sb`, `clam_mb`, `mil`, `transmil`

## Acknowledgement

This code is adapted from the CLAM model. The code for TransMIL model is copied from [their repo](https://github.com/szc19990412/TransMIL).

## License

This repository is licensed under the [GPLv3 License](./LICENSE). Note that this project is for non-commercial academic use only, in accordance with the licenses of the original models.

## References

Lu, Ming Y., et al. "Data-Efficient and Weakly Supervised Computational Pathology on Whole-Slide Images." Nature Biomedical Engineering, vol. 5, no. 6, 2021, pp. 555-570, Nature Publishing Group.

Shao, Zhuchen, et al. "Transmil: Transformer Based Correlated Multiple Instance Learning for Whole Slide Image Classification." Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 2136-2147.
