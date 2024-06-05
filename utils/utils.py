import os
import json
import numpy as np
import pandas as pd
import math
from itertools import islice
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Same as OViTANet utils.py

def get_tabular_data(args):
	
	with open(os.path.join(args.split_dir, "tabular_fs.json"), "r") as f:
		tabular_fs = json.load(f)
	split_data = [i.replace(" ", "") for i in args.tabular_data.split(",")]
	
	tabular_cols = []
	for v in split_data:
		tabular_cols.extend(tabular_fs[v])
			
	return sorted(tabular_cols)


class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)

def collate_MIL_survival(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)	
	label = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
	event_time = torch.FloatTensor([item[2] for item in batch])
	c = torch.FloatTensor([item[3] for item in batch])
	tabular = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
	case_id = np.array([item[5] for item in batch])
	
	return [img, label, event_time, c, tabular, case_id]

def collate_MIL(batch):
	img = torch.cat([item[0] for item in batch], dim = 0)
	label = torch.LongTensor(np.array([item[1] for item in batch]))
	ids = torch.LongTensor(np.array([item[2] for item in batch]))
	return [img, label, ids]

def get_simple_loader(dataset, batch_size=1):
	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
	return loader 

def get_split_loader(split_dataset, training = False, weighted = False, batch_size=1):
	"""
		return either the validation loader or training loader 
	"""
	
	collate = collate_MIL_survival

	kwargs = {'num_workers': 4} if device.type == "cuda" else {}
	
	if training:
		if weighted:
			weights = make_weights_for_balanced_classes_split(split_dataset)
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), drop_last = True if batch_size > 1 else False, collate_fn = collate, **kwargs)    
		else:
			loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), drop_last = True if batch_size > 1 else False, collate_fn = collate, **kwargs)
	else:
		loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)

	return loader

def print_network(net):
	num_params = 0
	num_params_train = 0
	print(net)
	
	for param in net.parameters():
		n = param.numel()
		num_params += n
		if param.requires_grad:
			num_params_train += n
	
	print('Total number of parameters: %d' % num_params)
	print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
	seed = 7, label_frac = 1.0, custom_test_ids = None):
	indices = np.arange(samples).astype(int)
	
	if custom_test_ids is not None:
		indices = np.setdiff1d(indices, custom_test_ids)

	np.random.seed(seed)
	for i in range(n_splits):
		all_val_ids = []
		all_test_ids = []
		sampled_train_ids = []
		
		if custom_test_ids is not None: # pre-built test split, do not need to sample
			all_test_ids.extend(custom_test_ids)

		for c in range(len(val_num)):
			possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
			remaining_ids = possible_indices

			if val_num[c] > 0:
				val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
				remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
				all_val_ids.extend(val_ids)

			if custom_test_ids is None and test_num[c] > 0: # sample test split

				test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
				remaining_ids = np.setdiff1d(remaining_ids, test_ids)
				all_test_ids.extend(test_ids)

			if label_frac == 1:
				sampled_train_ids.extend(remaining_ids)
			
			else:
				sample_num  = math.ceil(len(remaining_ids) * label_frac)
				slice_ids = np.arange(sample_num)
				sampled_train_ids.extend(remaining_ids[slice_ids])

		yield sorted(sampled_train_ids), sorted(all_val_ids), sorted(all_test_ids)


def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))                                           
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  

	return torch.DoubleTensor(weight)

def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)


def dfs_freeze(model):
	for name, child in model.named_children():
		for param in child.parameters():
			param.requires_grad = False
		dfs_freeze(child)


def dfs_unfreeze(model):
	for name, child in model.named_children():
		for param in child.parameters():
			param.requires_grad = True
		dfs_unfreeze(child)

def nth(iterator, n, default=None):
	if n is None:
		return collections.deque(iterator, maxlen=0)
	else:
		return next(islice(iterator,n, None), default)

def save_splits(split_datasets, column_keys, filename, boolean_style=False):
	splits = [split_datasets[i].slide_data['slide_id'] for i in range(len(split_datasets))]
	# [split_datasets[i].slide_data.to_csv(f"/home/ezgitwo/Desktop/sil{i}.csv") for i in range(len(split_datasets))]
	if not boolean_style:
		df = pd.concat(splits, ignore_index=True, axis=1)
		df.columns = column_keys
	else:
		df = pd.concat(splits, ignore_index = True, axis=0)
		index = df.values.tolist()
		one_hot = np.eye(len(split_datasets)).astype(bool)
		bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
		df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val', 'test'])

	df.to_csv(filename)

def get_custom_exp_code(args, feat_extractor=None):
	r"""
	Updates the argparse.NameSpace with a custom experiment code.

	Args:
		- args (NameSpace)

	Returns:
		- args (NameSpace)
	"""
	
	param_code = ''

	### Model Type
	param_code += args.model_type.upper()

	inputs = []
	if feat_extractor:
		param_code += "_" + feat_extractor
		inputs.append("path")
	if args.tabular_data:
		inputs.append("tab")
	args.mode = ("+").join(inputs)
	
	param_code += '_' + args.mode
	# param_code += '_%scls' % str(args.n_classes)
	
	### Updating
	args.param_code = param_code

	return args
