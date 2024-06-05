# Training for OS prediction
python main.py --results_dir results/os/ --split_dir tcga_ov_os --model_type clam_sb --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/
python main.py --results_dir results/os/ --split_dir tcga_ov_os --model_type clam_mb --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/
python main.py --results_dir results/os/ --split_dir tcga_ov_os --model_type mil --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/
python main.py --results_dir results/os/ --split_dir tcga_ov_os --model_type transmil --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/ --gc 2 --early_stopping 10 # Default parameters in their repo

# Training for DFS prediction
python main.py --results_dir results/dfs/ --split_dir tcga_ov_dfs --model_type clam_sb --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/
python main.py --results_dir results/dfs/ --split_dir tcga_ov_dfs --model_type clam_mb --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/
python main.py --results_dir results/dfs/ --split_dir tcga_ov_dfs --model_type mil --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/
python main.py --results_dir results/dfs/ --split_dir tcga_ov_dfs --model_type transmil --feats_dir /media/nfs/SURV/TCGA_OV/Feats1024/UNI/ --gc 2 --early_stopping 10 # Default parameters in their repo

# Evaluation for OS prediction
python eval.py --split_dir baskent_os --feats_dir /media/nfs/SURV/BasOVER/Feats1024/UNI/ --load_from ./results/os/CLAM_SB_UNI_path/run_s1/
python eval.py --split_dir baskent_os --feats_dir /media/nfs/SURV/BasOVER/Feats1024/UNI/ --load_from ./results/os/CLAM_MB_UNI_path/run_s1/
python eval.py --split_dir baskent_os --feats_dir /media/nfs/SURV/BasOVER/Feats1024/UNI/ --load_from ./results/os/MIL_UNI_path/run_s1/
python eval.py --split_dir baskent_os --feats_dir /media/nfs/SURV/BasOVER/Feats1024/UNI/ --load_from ./results/os/TRANSMIL_UNI_path/run_s1/

# Evaluation for DFS prediction
python eval.py --split_dir baskent_dfs --feats_dir /media/nfs/SURV/BasOVER/Feats1024/UNI/ --load_from ./results/dfs/CLAM_SB_UNI_path/run_s1/
python eval.py --split_dir baskent_dfs --feats_dir /media/nfs/SURV/BasOVER/Feats1024/UNI/ --load_from ./results/dfs/CLAM_MB_UNI_path/run_s1/
python eval.py --split_dir baskent_dfs --feats_dir /media/nfs/SURV/BasOVER/Feats1024/UNI/ --load_from ./results/dfs/MIL_UNI_path/run_s1/
python eval.py --split_dir baskent_dfs --feats_dir /media/nfs/SURV/BasOVER/Feats1024/UNI/ --load_from ./results/dfs/TRANSMIL_UNI_path/run_s1/
