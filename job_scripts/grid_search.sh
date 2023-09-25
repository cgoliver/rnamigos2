#!/bin/bash
#SBATCH --job-name="grid"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=5G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-96
#SBATCH --output grid-%a.out

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

						command="python experiments/train.py model.encoder.hidden_dim=64 \
									    model.decoder.in_dim=64 \
									    model.decoder.out_dim=166 \
									    model.batch_norm=${bn} \
									    model.droput=${drop} \
									    model.encoder.num_bases=${bases} \
									    train.target=native_fp \
									    train.loss=${lossfunc} \
									    model.use_pretrained=${pre} \
									    model.encoder.hidden_dim=64 \
									    model.pretrained_path=pretrained/R_iso_64/model.pth \
									    train.num_epochs=1000 \
									    train.early_stop=500 \
									    train.learning_rate=${lr} \
									    train.num_workers=4 \
									    name=fp_native_grid_${i} \
									    train.rnamigos1_split=0"
					        CMDARRAY+=("$command")
						i=$((i+1))
					done
				done
			done
		done
	done
done

for ((j=0; j < ${#CMDARRAY[@]};j++));
do
	eval $CMDARRAY[${SLURM_ARRAY_TASK_ID}]
done
