#!/bin/bash
#SBATCH --job-name="dockbigpockets"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output long-%a.out


cd ..
source .venv/bin/activate

# rnamigos1 no pretrain
python rnamigos/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16  model.decoder.out_dim=166 data.undirected=True train.target=native_fp train.use_rnamigos1_train=True name=retrain_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# add r_1 pretrain undirected
python rnamigos/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16  model.decoder.out_dim=166 data.undirected=True train.target=native_fp train.use_rnamigos1_train=True model.use_pretrained=True model.pretrained_path=pretrained/R_1_16_undirected/model.pth name=retrain_r1_undirected_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# rnamigos 1 architecture + new data
python rnamigos/train.py model.encoder.hidden_dim=16\
			    model.decoder.in_dim=16 \
			    model.decoder.out_dim=166 \
			    data.undirected=True \
			    train.target=native_fp \
			    train.use_rnamigos1_train=False \
			    name=whole_data_bigpockets_${SLURM_ARRAY_TASK_ID} \
			    train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# with BCE loss
python rnamigos/train.py model.encoder.hidden_dim=16 \
		            model.decoder.in_dim=16 \
			    model.decoder.out_dim=166 \
			    data.undirected=True \
			    train.target=native_fp \
			    train.loss=bce \
			    name=bce_loss_bigpockets_${SLURM_ARRAY_TASK_ID} \
			    train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# add directed edges
python rnamigos/train.py model.encoder.hidden_dim=16 \
		            model.decoder.in_dim=16 \
			    model.decoder.out_dim=166 \
			    data.undirected=False \
			    train.target=native_fp \
			    train.loss=bce \
			    name=directed_bigpockets_${SLURM_ARRAY_TASK_ID} \
			    train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# add R1 pretrain
python rnamigos/train.py model.encoder.hidden_dim=16 \
			    model.decoder.in_dim=16 \
			    model.decoder.out_dim=166 \
			    train.target=native_fp \
			    train.loss=l2 \
			    model.use_pretrained=True \
			    model.pretrained_path=pretrained/R_1_16/model.pth \
			    name=pre_r1_l2_bigpockets_${SLURM_ARRAY_TASK_ID} \
			    train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# R_graphlets
# python rnamigos/train.py model.encoder.hidden_dim=16 model.decoder.out_dim=166 train.target=native_fp loss=bce model.use_pretrained=True model.pretrained_path=pretrained/pretrained_model_Rgraphlets.pth name=pre_rgraphlets
# R_iso
python rnamigos/train.py model.encoder.hidden_dim=16 \
			    model.decoder.in_dim=16 \
			    model.decoder.out_dim=166 \
			    train.target=native_fp \
			    train.loss=l2 \
			    model.use_pretrained=True \
			    model.pretrained_path=pretrained/R_iso_16/model.pth \
			    name=pre_riso_small_l2_bigpockets_${SLURM_ARRAY_TASK_ID} \
			    train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# + Big model
python rnamigos/train.py model.encoder.hidden_dim=64 \
			    model.decoder.in_dim=64 \
			    model.decoder.out_dim=166 \
			    train.target=native_fp \
			    train.loss=l2 \
			    model.use_pretrained=True \
			    model.encoder.hidden_dim=64 \
			    model.pretrained_path=pretrained/R_iso_64/model.pth \
			    train.num_epochs=1000 \
			    train.early_stop=500 \
			    train.learning_rate=1e-6 \
			    name=pre_riso_big_slowslowlong_bigpockets_${SLURM_ARRAY_TASK_ID} \
			    train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

python rnamigos/train.py model.encoder.hidden_dim=64 \
			    model.decoder.in_dim=64 \
			    model.decoder.out_dim=166 \
			    train.target=native_fp \
			    train.loss=l2 \
			    model.use_pretrained=True \
			    model.decoder.num_layers=1 \
			    model.encoder.hidden_dim=64 \
			    model.pretrained_path=pretrained/R_iso_64/model.pth \
			    train.early_stop=50 \
			    train.learning_rate=1e-4 \
			    name=pre_riso_big_slow_shallowhead_bigpockets_${SLURM_ARRAY_TASK_ID} \
			    train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}

# add predict native training
 python rnamigos/train.py model.encoder.hidden_dim=64 \
			     model.decoder.in_dim=96 \
			     model.decoder.out_dim=1 \
			     train.target=is_native \
			     train.loss=bce \
			     model.use_pretrained=True \
			     model.encoder.hidden_dim=64 \
			     model.pretrained_path=pretrained/R_iso_64/model.pth \
			     name=train_native_bigpockets_${SLURM_ARRAY_TASK_ID} \
			     train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}
			      
# docking score target
python rnamigos/train.py model.encoder.hidden_dim=64 \
			    model.decoder.out_dim=166 \
			    train.target=dock \
			    train.loss=l2 \
			    model.use_pretrained=True \
			    model.decoder.in_dim=96 \
			    model.decoder.out_dim=1 \
			    model.pretrained_path=pretrained/R_iso_64/model.pth \
			    name=train_dock_bibpockets_${SLURM_ARRAY_TASK_ID} \
			    train.rnamigos1_split=${SLURM_ARRAY_TASK_ID} \
			    model.decoder.activation=none \
			    train.num_workers=4
