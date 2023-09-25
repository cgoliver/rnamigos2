#!/bin/bash
#SBATCH --job-name="grid"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output grid.out

cd ..
source .venv/bin/activate

CMDARRAY=()
i=0

for lossfunc in bce l2;
do
	for drop in 0.0 0.3;
	do
		for bn in true false;
		do
			for pre in true false;
			do
				for lr in 1e-4 1e-6;
				do	
					for bases in -1 4 8;
					do

						python experiments/train.py model.encoder.hidden_dim=64 \
									    model.decoder.in_dim=64 \
									    model.decoder.out_dim=166 \
									    model.batch_norm=${bn} \
									    model.dropout=${drop} \
									    model.encoder.num_bases=${bases} \
									    model.use_pretrained=${pre} \
									    model.encoder.hidden_dim=64 \
									    model.pretrained_path=pretrained/R_iso_64/model.pth \
									    model.decoder.activation=sigmoid \
									    train.target=native_fp \
									    train.loss=${lossfunc} \
									    train.num_epochs=1000 \
									    train.early_stop=500 \
									    train.learning_rate=${lr} \
									    train.num_workers=4 \
									    train.rnamigos1_split=0 \
									    name=fp_native_grid-${bn}-${drop}-${bases}-${lossfunc}-${pre}-${lr}
					done
				done
			done
		done
	done
done
