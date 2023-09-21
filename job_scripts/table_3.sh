#!/bin/bash
#SBATCH --job-name="rnamigos_train"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:h100_pcie_1g.10gb:1
#SBATCH --partition=p.hpcl91
#SBATCH --array=0-9


cd ..
source .venv/bin/activate

# python experiments/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16  model.decoder.out_dim=166 data.undirected=True train.target=native_fp train.use_rnamigos1_train=True name=retrain_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}
# rnamigos 1 architecture + new data
python experiments/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16 model.decoder.out_dim=166 data.undirected=True train.target=native_fp train.use_rnamigos1_train=False name=whole_data_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# with BCE loss
python experiments/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16 model.decoder.out_dim=166 data.undirected=True train.target=native_fp train.loss=bce name=bce_loss_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# add directed edges
python experiments/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16 model.decoder.out_dim=166 data.undirected=False train.target=native_fp train.loss=bce name=directed_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# add R1 pretrain
python experiments/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16 model.decoder.out_dim=166 train.target=native_fp train.loss=bce model.use_pretrained=True model.pretrained_path=pretrained/R_1_16/model.pth name=pre_r1_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# R_graphlets
# python experiments/train.py model.encoder.hidden_dim=16 model.decoder.out_dim=166 train.target=native_fp loss=bce model.use_pretrained=True model.pretrained_path=pretrained/pretrained_model_Rgraphlets.pth name=pre_rgraphlets
# R_iso
python experiments/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16 model.decoder.out_dim=166 train.target=native_fp train.loss=bce model.use_pretrained=True model.pretrained_path=pretrained/R_iso_16/model..pth name=pre_riso_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# + Big model
python experiments/train.py model.encoder.hidden_dim=64 model.decoder.in_dim=64 model.decoder.out_dim=166 train.target=native_fp train.loss=bce model.use_pretrained=True model.encoder.hidden_dim=64 model.pretrained_path=pretrained/R_iso_64/model.pth name=pre_riso_big_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# add predict native training
python experiments/train.py model.encoder.hidden_dim=64 model.decoder.in_dim=64 model.decoder.out_dim=166 train.target=is_native train.loss=bce model.use_pretrained=True model.encoder.hidden_dim=64 model.pretrained_path=pretrained/R_iso_64/model.pth name=train_native_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# docking score target
python experiments/train.py model.encoder.hidden_dim=64 model.decoder.out_dim=166 train.target=dock train.loss=bce model.use_pretrained=True model.decoder.in_dim=96 model.decoder.out_dim=1 model.pretrained_path=pretrained/R_iso_64/model.pth name=train_dock_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

