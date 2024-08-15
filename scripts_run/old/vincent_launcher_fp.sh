#!/bin/bash

CMDARRAY=()

for split in {0..9};
do
  #  FIRST NO PRE, UNDIRECTED
  python_cmd="python rnamigos/train.py
                    data.undirected=True
                    model.decoder.in_dim=16
                    model.decoder.out_dim=166
                    model.encoder.num_bases=null
                    model.encoder.hidden_dim=16
                    model.decoder.activation=sigmoid
                    train.target=native_fp
                    train.loss=bce
                    train.num_epochs=1000
                    train.early_stop=100
                    train.learning_rate=1e-3
                    train.num_workers=0
                    train.rnamigos1_split=$split
                    train.use_rnamigos1_train=True
                    device=cpu
                    name=rnamigos1_repro_real_$split"
  python_cmd=$(echo $python_cmd) # to replace newlines
  CMDARRAY+=("$python_cmd")

  # directed migos1
  python_cmd="python rnamigos/train.py
                        model.decoder.in_dim=16
										    model.decoder.out_dim=166
										    model.encoder.num_bases=null
										    model.encoder.hidden_dim=16
										    model.decoder.activation=sigmoid
										    train.target=native_fp
										    train.loss=bce
										    train.num_epochs=1000
										    train.early_stop=100
										    train.learning_rate=1e-3
										    train.num_workers=0
										    train.rnamigos1_split=$split
										    train.use_rnamigos1_train=True
										    device=cpu
										    name=rnamigos1_repro_$split"
	python_cmd=$(echo $python_cmd) # to replace newlines
  CMDARRAY+=("$python_cmd")


	for dim in 16 64;
    do
    python_cmd="python rnamigos/train.py
                      model.encoder.hidden_dim=16
                      model.decoder.in_dim=${dim}
                      model.decoder.out_dim=166
                      model.encoder.hidden_dim=${dim}
                      model.decoder.activation=sigmoid
                      train.target=native_fp
                      train.loss=bce
                      train.num_epochs=1000
                      train.early_stop=100
                      train.learning_rate=1e-3
                      train.num_workers=0
                      train.rnamigos1_split=${split}
                      train.use_rnamigos1_train=False
                      device=cpu
                      name=rnamigos2_dim${dim}_${split}"
    python_cmd=$(echo $python_cmd) # to replace newlines
    CMDARRAY+=("$python_cmd")
    done
done

N_JOBS=50
for ((i = 0; i < ${#CMDARRAY[@]}; i++))
do
    eval "${CMDARRAY[$i]}" &
    [ $(( i+1 % N_JOBS )) -eq 0 ]  && wait
done


