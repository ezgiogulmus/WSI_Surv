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
from wsisurv.datasets.dataset_survival import MIL_Survival_Dataset
from wsisurv.utils.file_utils import save_pkl
from wsisurv.utils.core_utils import train
from wsisurv.utils.utils import check_directories
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(args=None):
	if args is None:
		args = setup_argparse()
	args.mode = "path"
	seed_torch(args.seed)

	args = check_directories(args)
		
	os.makedirs(args.results_dir, exist_ok=True)
	if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
		print("Exp Code <%s> already exists! Exiting script." % args.run_name)
		sys.exit()

	settings = vars(args)
	print("Saving to ", args.results_dir)
	with open(args.results_dir + '/experiment.json', 'w') as f:
		json.dump(settings, f, indent=4)
	
	print("################# Settings ###################")
	for key, val in settings.items():
		print("{}:  {}".format(key, val)) 

	print("Loading all the data ...")
	df = pd.read_csv(args.csv_path, compression="zip" if ".zip" in args.csv_path else None)
	print("Total number of cases: {} | slides: {}" .format(len(df["case_id"].unique()), len(df)))

	dataset = MIL_Survival_Dataset(
		df=df,
		data_dir=args.feats_dir,
		mode= args.mode,
		print_info=True,
		n_bins=args.n_classes
	)

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
		val_results_pkl_path = os.path.join(args.results_dir, 'latest_val_results_split{}.pkl'.format(i))
		test_results_pkl_path = os.path.join(args.results_dir, 'latest_test_results_split{}.pkl'.format(i))
		if os.path.isfile(test_results_pkl_path):
			print("Skipping Split %d" % i)
			continue

		### Gets the Train + Val Dataset Loader.
		datasets, train_stats = dataset.return_splits(os.path.join(args.split_dir, f"splits_{i}.csv"))
		if train_stats is not None:
			train_stats.to_csv(os.path.join(args.results_dir, f'train_stats_{i}.csv'))
		
		log, val_latest, test_latest = train(datasets, i, args)
		
		if results is None:
			results = {k: [] for k in log.keys()}
		
		for k in log.keys():
			results[k].append(log[k])
		
		save_pkl(val_results_pkl_path, val_latest)
		if test_latest != None:
			save_pkl(test_results_pkl_path, test_latest)
		end = timer()
		print('Fold %d Time: %f seconds' % (i, end - start))
	
		pd.DataFrame(results).to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))


def setup_argparse():
	### Data 
	parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
	
	parser.add_argument('--data_name',   type=str, default=None)
	parser.add_argument('--feats_dir',   type=str, default=None)
	parser.add_argument('--dataset_dir', type=str, default="./datasets_csv")
	parser.add_argument('--results_dir', type=str, default='./results', help='Results directory (Default: ./results)')
	parser.add_argument('--split_dir', type=str, default="./splits", help='Split directory (Default: ./splits)')

	### Experiment
	parser.add_argument('--run_name',      type=str, default='run')
	parser.add_argument('--run_config_file',      type=str, default=None)
	parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
	parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
	parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
	parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
	parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
	parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')

	### Model Parameters
	parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil', "transmil"], default='clam_sb',  help='type of model (default: clam_sb, clam w/ single attention branch)')
	parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
	parser.add_argument('--drop_out',        default=.25, type=float, help='Enable dropout (p=0.25)')
	parser.add_argument('--n_classes', type=int, default=4)
	parser.add_argument('--surv_model', default="discrete", choices=["cont", "discrete"])
	
	### CLAM
	parser.add_argument('--no_inst_cluster', action='store_true', default=False, help='disable instance-level clustering')
	parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default="svm", help='instance-level clustering loss function (default: None)')
	parser.add_argument('--subtyping', action='store_false', default=True, help='subtyping problem')
	parser.add_argument('--bag_weight', type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
	parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')

	### Training Parameters
	parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
	parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
	parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
	parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum number of epochs to train (default: 20)')
	parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
	parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
	parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
	parser.add_argument('--lambda_reg',      type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
	parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')

	parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
	parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
	parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')

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

			
if __name__ == "__main__" and (__package__ is None or __package__ == ''):
	script_dir = os.path.dirname(os.path.abspath(__file__))
	sys.path.append(os.path.dirname(script_dir))
	
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
		results = run(args)
		end = timer()
		print("finished!")
		print("end script")
		print('Script Time: %f seconds' % (end - start))
	else:
		start = timer()
		results = run()
		end = timer()
		print("finished!")
		print("end script")
		print('Script Time: %f seconds' % (end - start))
	