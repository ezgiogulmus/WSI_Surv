# CLAM for Discrete-Time Survival Prediction

This repository contains an adaptation of the CLAM (Clustering-constrained Attention Multiple Instance Learning) model for discrete-time survival prediction tasks. 

## Requirements

- Python 3.10
- PyTorch 2.3

## Installation

To set up the environment and install the necessary dependencies, run the following commands:

```bash
git clone https://github.com/ezgiogulmus/CLAM_Surv.git
cd CLAM_Surv
conda env create -n clam_surv --file env.yml
conda activate clam_surv
```

## Usage

First, extract patch coordinates using the CLAM library available at [CLAM GitHub](https://github.com/Mahmoodlab/CLAM). Then, run the following command:

```bash
python main.py --split_dir name_of_the_split_folder --model_type clam_sb --feats_dir path/to/features_directory
```

- `model_type`: Options are `clam_sb`, `clam_mb`, `mil`

## Acknowledgement

This code is adapted from the CLAM model.

## License

This repository is licensed under the [GPLv3 License](./LICENSE). Note that this project is for non-commercial academic use only, in accordance with the licenses of the original model.

## References

Lu, Ming Y., et al. "Data-Efficient and Weakly Supervised Computational Pathology on Whole-Slide Images." Nature Biomedical Engineering, vol. 5, no. 6, 2021, pp. 555-570, Nature Publishing Group.
