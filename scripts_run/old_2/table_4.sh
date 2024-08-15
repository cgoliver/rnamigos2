#!/bin/bash
#SBATCH --job-name="table4"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output long-%a.out


cd ..
source .venv/bin/activate

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
			    train.learning_rate=1e-4 \
			    name=pre_riso_big_slow_nokfold_bigpockets

# add predict native training
python rnamigos/train.py model.encoder.hidden_dim=64 \
		            model.decoder.in_dim=96 \
			    model.decoder.out_dim=1 \
			    train.learning_rate=1e-4 \
			    train.target=is_native \
			    train.loss=bce \
			    model.use_pretrained=True \
			    model.encoder.hidden_dim=64 \
			    model.pretrained_path=pretrained/R_iso_64/model.pth\
			    name=train_native_nokfold_bigpockets

# docking score target
python rnamigos/train.py model.encoder.hidden_dim=64 \
			    model.decoder.out_dim=166 \
			    train.target=dock \
			    train.loss=l2 \
			    model.use_pretrained=True \
			    model.decoder.in_dim=96 \
			    model.decoder.out_dim=1 \
			    model.pretrained_path=pretrained/R_iso_64/model.pth \
			    train.learning_rate=1e-4 \
			    name=train_dock_nokfold_bigpockets \
			    model.decoder.activation=none train.num_workers=4
