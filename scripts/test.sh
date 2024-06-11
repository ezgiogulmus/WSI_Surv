#!/bin/bash

data_name="tcga_ov_os"
cancer_type="TCGA_OV"
omics="rna"
output_file="./scripts/error_output.txt"

> "$output_file"

# Function to run a command and capture output on error
run_command() {
  local cmd="$1"
  echo "Running: $cmd"
  eval "$cmd" || (echo "Command failed, capturing output..." && eval "$cmd >> $output_file 2>&1")
}

run_command "CUDA_VISIBLE_DEVICES=0 python run_wsisurv.py --results_dir results/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type mil --weighted_sample --gc 1 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --drop_out 0.25 --inst_loss svm"
run_command "CUDA_VISIBLE_DEVICES=0 python run_wsisurv.py --results_dir results/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type clam_mb --weighted_sample --gc 1 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --drop_out 0.25 --inst_loss svm"
run_command "CUDA_VISIBLE_DEVICES=0 python run_wsisurv.py --results_dir results/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type clam_sb --weighted_sample --gc 1 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 20 --drop_out 0.25 --inst_loss svm"
run_command "CUDA_VISIBLE_DEVICES=0 python run_wsisurv.py --results_dir results/${data_name} --data_name ${data_name} --feats_dir /media/nfs/SURV/${cancer_type}/Feats1024/UNI/ --model_type transmil --gc 2 --lr 2e-4 --reg 1e-5 --max_epochs 1 --k 1 --early_stopping 10 --early_stopping 20"