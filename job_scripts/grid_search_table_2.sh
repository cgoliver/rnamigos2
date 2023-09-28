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



for split in {0..9};
do
	# rnamigos 1 grid. includes original setting + R_1 + potential enhancements
	for dropout in 0 0.3;
	do
		for dim in 16 64;
		do
			for undirected in true false;
			do
				for smalldata in true false;
				do
					for pre in true false;
					do
						CMDARRAY+=("python experiments/train.py model.encoder.hidden_dim=${dim}
										    data.undirected=${undirected}
										    model.decoder.in_dim=${dim}
										    model.decoder.out_dim=166
										    model.batch_norm=False
										    model.dropout=${dropout}
										    model.encoder.num_bases=null
										    model.use_pretrained=${pre}
										    model.pretrained_path=pretrained/R_1_${dim}/model.pth
										    model.encoder.hidden_dim=${dim}
										    model.decoder.activation=sigmoid 
										    train.target=native_fp 
										    train.loss=bce
										    train.num_epochs=1000 
										    train.early_stop=100 
										    train.learning_rate=1e-3 
										    train.num_workers=4 
										    train.rnamigos1_split=${split}
										    train.use_rnamigos1_train=${smalldata}
										    device=cpu
										    name=table2-rnamigos1-nopre-smalldata_${smalldata}-dim-${dim}-undirected_{undirected}-dropout-{droput}-split_${split}"
									    )
					    done
				    done
			    done
		    done
	    done

	# the pretrains
	for simfunc in R_1 R_iso hungarian;
	do
		for pretrain_weight in 0 0.5;
		do
			CMDARRAY+=("python experiments/train.py 
						    data.undirected=False
						    data.use_rnamigos1
				            	    model.encoder.hidden_dim=64 
						    model.decoder.in_dim=64 
						    model.decoder.out_dim=166
						    model.batch_norm=False
						    model.dropout=0.3
						    model.encoder.num_bases=null
						    model.use_pretrained=true
						    model.encoder.hidden_dim=64
						    model.pretrained_path=pretrained/${simfunc}_64/model.pth
						    model.decoder.activation=sigmoid 
						    train.pretrain_weight=${pretrain_weight}
						    train.target=native_fp 
						    train.loss=bce
						    train.num_epochs=1000 
						    train.early_stop=100 
						    train.learning_rate=1e-3 
						    train.num_workers=4 
						    train.rnamigos1_split=${split}
						    train.use_rnamigos1_train=False
						    device=cpu
						    name=table2-pretrains-simfunc_${simfunc}-pw_${pretrain_weight}-split_{split}"
					    )



	# is_native
	for pre in true false;
	do
		CMDARRAY+=("python experiments/train.py 
					    data.undirected=False
					    model.encoder.hidden_dim=64 
					    model.decoder.in_dim=96
					    model.decoder.out_dim=1
					    model.batch_norm=False
					    model.dropout=0.3
					    model.encoder.num_bases=null
					    model.use_pretrained=${pre}
					    model.pretrained_path=pretrained/R_iso_64/model.pth
					    model.decoder.activation=sigmoid 
					    train.pretrain_weight=0.5
					    train.target=is_native
					    train.loss=bce
					    train.num_epochs=1000 
					    train.early_stop=100 
					    train.learning_rate=1e-3 
					    train.num_workers=4 
					    train.rnamigos1_split=${split}
					    train.use_rnamigos1_train=False
					    device=cpu
					    name=table2-pretrains-target_isnative-pre_${pre}-split_${split}"
					    )


	# dock
	for pre in true false;
	do
		CMDARRAY+=("python experiments/train.py 
					    data.undirected=False
					    model.encoder.hidden_dim=64 
					    model.decoder.in_dim=96
					    model.decoder.out_dim=1
					    model.batch_norm=False
					    model.dropout=0.3
					    model.encoder.num_bases=null
					    model.use_pretrained=${pre}
					    model.pretrained_path=pretrained/R_iso_64/model.pth
					    model.decoder.activation=sigmoid 
					    train.pretrain_weight=0.5
					    train.target=is_native
					    train.loss=bce
					    train.num_epochs=1000 
					    train.early_stop=100 
					    train.learning_rate=1e-3 
					    train.num_workers=4 
					    train.rnamigos1_split=${split}
					    train.use_rnamigos1_train=False
					    device=cpu
					    name=table2-pretrains-target_dock-pre_${pre}-split_${split}"
					    )
	done
done

eval ${CMDARRAY[${SLURM_ARRAY_TASK_ID}]}
