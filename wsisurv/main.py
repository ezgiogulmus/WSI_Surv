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

def run(args):
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

