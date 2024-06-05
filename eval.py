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
from utils.core_utils import eval_model
from utils.utils import get_custom_exp_code, get_tabular_data

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(eval_args): 
	
	eval_args.results_dir = eval_args.load_from
	feats_dir = eval_args.feats_dir
	
	split_dir = os.path.join('./splits', eval_args.split_name)
	print("split_dir", split_dir)
	assert os.path.isdir(split_dir), "Incorrect the split directory"

	csv_path = f"./datasets_csv/{eval_args.split_name}.csv"
	print("csv_path", csv_path)
	assert os.path.isfile(csv_path), "Incorrect csv file path"

	load_from = eval_args.load_from
	test_all = eval_args.test_all
	
	with open(os.path.join(eval_args.results_dir, "experiment.json"), "r") as f:
		config = json.load(f)

	parser = argparse.ArgumentParser()
	parser.add_argument('--load_from', default=load_from)
	parser.add_argument('--test_all', default=test_all, action="store_false")
	parser.add_argument('--split_name', default=eval_args.split_name)
	for k, v in config.items():
		parser.add_argument('--' + k, default=v, type=type(v))
	
	args = parser.parse_args()
	
	survival_time_list = pd.read_csv(config["csv_path"])["survival_months"].values
	# assert args.nb_tabular_data < 2, "Tabular data integration is not implemented other than age-only scenerio."
	tabular_cols = []
	if args.nb_tabular_data > 0:
		# tabular_cols = ["age"]
		tabular_cols = get_tabular_data(args)	

	args.feats_dir = feats_dir
	args.split_dir = split_dir
	args.load_from = load_from
	# args.cv = os.path.basename(args.load_from).split("_")[1]
	args.dataname = eval_args.split_name
	print("Experiment Name:", args.run_name)
	
	seed_torch(args.seed)
	
	if f'{args.dataname}_eval_summary.csv' in os.listdir(eval_args.results_dir):
		print("Exp Code <%s> already exists! Exiting script." % args.run_name)
		sys.exit()

	settings = vars(args)
	print('\nLoad Dataset')
	df = pd.read_csv(csv_path)
	gen_data = np.unique([i.split("_")[-1] for i in tabular_cols if i.split("_")[-1] in ["pro", "rna", "rnz", "dna", "mut", "cnv"]])
	if len(gen_data) > 0:
		for g in gen_data:
			gen_df = pd.read_csv(f"./datasets_csv/{args.dataname}_{g}.csv.zip", compression="zip")
			df = pd.merge(df, gen_df, on='case_id')#, how="outer")
	df = df.reset_index(drop=True).drop(df.index[df["event"].isna()]).reset_index(drop=True)
	surv_dataset = MIL_Survival_Dataset(
		df=df,
		data_dir= args.feats_dir,
		mode= args.mode,
		print_info = True,
		n_bins=args.n_classes,
		indep_vars=tabular_cols,
		survival_time_list=survival_time_list
	)

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
	folds = np.arange(start, end)
	for cv in folds:
		
		if test_all:
			dataset = surv_dataset.return_splits(return_all=True, stats_path=os.path.join(eval_args.results_dir, f'train_stats_{cv}.csv'))
		else:
			assert args.dataname in args.split_dir, "Testing is only possible for the same dataset."
			_, _, dataset = surv_dataset.return_splits(cv)
		
		result_latest = eval_model(dataset, eval_args.results_dir, args, cv)
	
		if os.path.isfile(os.path.join(eval_args.results_dir, f'{args.dataname}_eval_summary.csv')):
			results_df = pd.read_csv(os.path.join(eval_args.results_dir, f'{args.dataname}_eval_summary.csv'))
			results_df = results_df.append(result_latest, ignore_index=True)
			results_df.to_csv(os.path.join(eval_args.results_dir, f'{args.dataname}_eval_summary.csv'), index=False)
		else:
			pd.DataFrame(result_latest, index=[0], dtype=float).to_csv(os.path.join(eval_args.results_dir, f'{args.dataname}_eval_summary.csv'), index=False)


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
	parser = argparse.ArgumentParser()
	parser.add_argument('--load_from',      type=str, default=None)
	parser.add_argument('--split_name',   type=str, default=None)
	parser.add_argument('--feats_dir',   type=str, default=None)
	parser.add_argument('--test_all',   action='store_false', default=True)
	eval_args = parser.parse_args()
	
	start = timer()
	results = main(eval_args)
	end = timer()
	print("finished!")
	print("end script")
	print('Script Time: %f seconds' % (end - start))
	