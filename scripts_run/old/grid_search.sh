#!/bin/bash
#SBATCH --job-name="allgrid"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=12:00:00
#SBATCH --output test_%a.out
#SBATCH --array=0-100%20

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
				for bases in none 4 8;
				do

					CMDARRAY+=("python rnamigos/train.py model.encoder.hidden_dim=64
								    model.decoder.in_dim=64 
								    model.decoder.out_dim=166
								    model.batch_norm=${bn}
								    model.dropout=${drop}
								    model.encoder.num_bases=${bases}
								    model.use_pretrained=${pre}
								    model.encoder.hidden_dim=64
								    model.pretrained_path=pretrained/R_iso_64/model.pth
								    model.decoder.activation=sigmoid 
								    train.target=native_fp 
								    train.loss=${lossfunc} 
								    train.num_epochs=1000 
								    train.early_stop=50 
								    train.learning_rate=1e-4 
								    train.num_workers=4 
								    train.rnamigos1_split=0 
								    device=cpu
								    name=fp_native_grid2-${bn}-${drop}-${bases}-${lossfunc}-${pre}"
							    )
				done
			done
		done
	done
done

eval ${CMDARRAY[${SLURM_ARRAY_TASK_ID}]}
