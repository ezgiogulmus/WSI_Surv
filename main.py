from __future__ import print_function
import argparse

import os
import sys
import json
from timeit import default_timer as timer
import numpy as np
import pandas as pd
import torch

### Internal Imports
from datasets.dataset_survival import MIL_Survival_Dataset
from utils.file_utils import save_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code, get_tabular_data

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args=None):
	if args is None:
		args = setup_argparse()
	
	feat_extractor = None
	if args.feats_dir:
		feat_extractor = args.feats_dir.split('/')[-1] if len(args.feats_dir.split('/')[-1]) > 0 else args.feats_dir.split('/')[-2]
		if feat_extractor == "RESNET50":
			args.path_input_dim = 2048 
		elif feat_extractor in ["PLIP", "CONCH"]:
			args.path_input_dim = 512 
		elif feat_extractor == "UNI":
			args.path_input_dim = 1024
		else:
			args.path_input_dim = 768

	args = get_custom_exp_code(args, feat_extractor)
		
	print("Experiment Name:", args.run_name)
	seed_torch(args.seed)
	
	split_name = args.split_dir
	args.split_dir = os.path.join('./splits', split_name)
	print("split_dir", args.split_dir)
	assert os.path.isdir(args.split_dir)

	args.results_dir = os.path.join(args.results_dir, args.param_code, args.run_name + '_s{}'.format(args.seed))

	settings = vars(args)
	print('\nLoad Dataset')
	
	tabular_cols = get_tabular_data(args) if args.tabular_data not in ["None", "none", None] else []
	args.nb_tabular_data = len(tabular_cols)
	
	if args.nb_tabular_data > 0:
		suffix = ""
		if args.feats_dir not in [None, "None", "none"]:
			suffix += "_"+args.mm_fusion_type+","+args.mm_fusion
		suffix += "_"+args.tabular_data
		args.results_dir += suffix
	
	os.makedirs(args.results_dir, exist_ok=True)
	if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
		print("Exp Code <%s> already exists! Exiting script." % args.run_name)
		sys.exit()
	
	if args.csv_path is None:
		args.csv_path = f"{args.dataset_path}/"+split_name+".csv"
	print("Loading all the data ...")
	df = pd.read_csv(args.csv_path)

	gen_data = np.unique([i.split("_")[-1] for i in tabular_cols if i.split("_")[-1] in ["pro", "rna", "rnz", "dna", "mut", "cnv"]])
	if len(gen_data) > 0:
		for g in gen_data:
			gen_df = pd.read_csv(f"{args.dataset_path}/{split_name}_{g}.csv.zip", compression="zip")
			df = pd.merge(df, gen_df, on='case_id')#, how="outer")
	df = df.reset_index(drop=True).drop(df.index[df["event"].isna()]).reset_index(drop=True)
	# assert df.isna().any().any() == False, "There are NaN values in the dataset."
	print("Successfully loaded.")
	dataset = MIL_Survival_Dataset(
		df=df,
		data_dir= args.feats_dir,
		mode= args.mode,
		print_info = True,
		n_bins=args.n_classes,
		indep_vars=tabular_cols
	)

	print("Saving to ", args.results_dir)
	with open(args.results_dir + '/experiment.json', 'w') as f:
		json.dump(settings, f, indent=4)

	print("################# Settings ###################")
	for key, val in settings.items():
		print("{}:  {}".format(key, val))  
		
	if args.k_start == -1:
		start = 0
	else:
		start = args.k_start
	if args.k_end == -1:
		end = args.k
	else:
		end = args.k_end

	results = None
	folds = np.arange(start, end)

	### Start 5-Fold CV Evaluation.
	for i in folds:
		start = timer()
		seed_torch(args.seed)
		results_pkl_path = os.path.join(args.results_dir, 'split_latest_test_{}_results.pkl'.format(i))
		if os.path.isfile(results_pkl_path):
			print("Skipping Split %d" % i)
			continue

		### Gets the Train + Val Dataset Loader.
		datasets, train_stats = dataset.return_splits(os.path.join(args.split_dir, f"splits_{i}.csv"))
		train_stats.to_csv(os.path.join(args.results_dir, f'train_stats_{i}.csv'))
		
		log, val_latest, test_latest = train(datasets, i, args)
		
		if results is None:
			results = {k: [] for k in log.keys()}
		
		for k in log.keys():
			results[k].append(log[k])

		### Write Results for Each Split to PKL
		if test_latest != None:
			save_pkl(results_pkl_path, test_latest)
		end = timer()
		print('Fold %d Time: %f seconds' % (i, end - start))
	
		pd.DataFrame(results).to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))


def setup_argparse():
	parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
	parser.add_argument('--run_name',      type=str, default='run')
	parser.add_argument('--csv_path',   type=str, default=None)
	parser.add_argument('--dataset_path', type=str, default="./datasets_csv")
	parser.add_argument('--run_config_file',      type=str, default=None)
	
	parser.add_argument('--feats_dir',   type=str, default=None)
	parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
	parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
	parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
	parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
	parser.add_argument('--results_dir',     type=str, default='./results', help='Results directory (Default: ./results)')

	parser.add_argument('--split_dir',       type=str, default="tcga_ov_os", help='Which cancer type within ./splits/<which_splits> to use for training.')
	parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
	parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')

	### Model Parameters.
	parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce', help='slide-level classification loss function (default: ce)')
	parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', "transmil"], default='clam_sb',  help='type of model (default: clam_sb, clam w/ single attention branch)')
	parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
	parser.add_argument('--drop_out',        default=.25, type=float, help='Enable dropout (p=0.25)')
	parser.add_argument('--no_inst_cluster', action='store_true', default=False, help='disable instance-level clustering')
	parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default="svm", help='instance-level clustering loss function (default: None)')
	parser.add_argument('--subtyping', action='store_false', default=True, help='subtyping problem')
	parser.add_argument('--bag_weight', type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
	parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

	parser.add_argument('--n_classes', type=int, default=4)
	parser.add_argument('--surv_model', default="discrete", choices=["cont", "discrete"])
	parser.add_argument('--tabular_data', default=None)

	parser.add_argument('--mm_fusion',        type=str, choices=["crossatt", "concat", "adaptive", "multiply", "bilinear", "lrbilinear", None], default=None)
	parser.add_argument('--mm_fusion_type',   type=str, choices=["early", "mid", "late", None], default=None)

	### Optimizer Parameters + Survival Loss Function
	parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
	parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
	parser.add_argument('--gc',              type=int, default=1, help='Gradient Accumulation Step during training (Gradients are calculated for every 256 patients)')
	parser.add_argument('--max_epochs',      type=int, default=200, help='Maximum number of epochs to train')
	
	parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate')
	parser.add_argument('--train_fraction',      type=float, default=1., help='fraction of training patches')
	parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay')
	
	parser.add_argument('--weighted_sample', action='store_false', default=True, help='Enable weighted sampling')
	parser.add_argument('--early_stopping',  default=20, type=int, help='Enable early stopping')
	parser.add_argument('--bootstrapping', action='store_true', default=False)


	args = parser.parse_args()
	return args


### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

			

if __name__ == "__main__":
	args = setup_argparse()
	if args.run_config_file:
		new_run_name = args.run_name
		results_dir = args.results_dir
		feats_dir = args.feats_dir
		cv_fold = args.k
		max_epochs = args.max_epochs
		with open(args.run_config_file, "r") as f:
			config = json.load(f)
		
		parser = argparse.ArgumentParser()
		parser.add_argument("--run_config_file")
		for k, v in config.items():
			if k != "run_config_file":
				parser.add_argument('--' + k, default=v, type=type(v))
		args = parser.parse_args()
		args.run_name = new_run_name
		args.feats_dir = feats_dir
		args.results_dir = results_dir
		args.k = cv_fold
		args.max_epochs = max_epochs
		args.split_dir = args.split_dir.split("/")[-1]
		start = timer()
		results = main(args)
		end = timer()
		print("finished!")
		print("end script")
		print('Script Time: %f seconds' % (end - start))
	else:
		start = timer()
		results = main()
		end = timer()
		print("finished!")
		print("end script")
		print('Script Time: %f seconds' % (end - start))
	