#!/bin/bash
#SBATCH --job-name="migos1"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=12:00:00
#SBATCH --output migos1_%a.out
#SBATCH --array=0-20

cd ..
source .venv/bin/activate

CMDARRAY=()

# Rnamigos 1
#
#
for split in {0..9};
do
	CMDARRAY+=("python rnamigos/train.py model.encoder.hidden_dim=16
						    data.undirected=true
						    model.decoder.in_dim=16
						    model.decoder.out_dim=166
						    model.batch_norm=false
						    model.dropout=0
						    model.encoder.num_bases=null
						    model.use_pretrained=false
						    model.encoder.hidden_dim=16
						    model.decoder.activation=sigmoid 
						    train.target=native_fp 
						    train.loss=bce
						    train.num_epochs=1000 
						    train.early_stop=100 
						    train.learning_rate=1e-3 
						    train.num_workers=0 
						    train.rnamigos1_split=${split}
						    train.use_rnamigos1_train=true
						    device=cpu
						    name=table2-rnamigos1-nopre-split_${split}"
					    )

	CMDARRAY+=("python rnamigos/train.py model.encoder.hidden_dim=16
						    data.undirected=true
						    model.decoder.in_dim=16
						    model.decoder.out_dim=166
						    model.batch_norm=False
						    model.dropout=0
						    model.encoder.num_bases=null
						    model.use_pretrained=True
						    model.encoder.hidden_dim=16
						    model.decoder.activation=sigmoid 
						    model.pretrained_path=pretrained/R_1_16/model.pth
						    train.target=native_fp 
						    train.loss=bce
						    train.num_epochs=1000 
						    train.early_stop=100 
						    train.learning_rate=1e-3 
						    train.num_workers=0 
						    train.rnamigos1_split=${split}
						    train.use_rnamigos1_train=true
						    device=cpu
						    name=table2-rnamigos1-pre-split_${split}"
					    )
done

eval ${CMDARRAY[${SLURM_ARRAY_TASK_ID}]}
