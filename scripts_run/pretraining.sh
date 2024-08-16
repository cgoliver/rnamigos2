#!/bin/bash

CMDARRAY=()

# First undirected
python_cmd="python rnamigos/pretrain.py
              model.dropout=0.3
              model.encoder.num_bases=null
              model.encoder.hidden_dim=16
              model.encoder.subset_pocket_nodes=True
              data.undirected=True
              simfunc=R_1
              name=R_1_undirected"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")


for simfunc in R_1 R_iso hungarian;
do
	for dim in 16 64;
    do
    python_cmd="python rnamigos/pretrain.py
                  model.dropout=0.3
                  model.encoder.hidden_dim=${dim}
                  model.encoder.subset_pocket_nodes=True
                  simfunc=${simfunc}
                  name=${simfunc}_${dim}"
    python_cmd=$(echo $python_cmd) # to replace newlines
    CMDARRAY+=("$python_cmd")
    done
done


#echo "${CMDARRAY[0]}"

N_JOBS=4
for ((i = 0; i < ${#CMDARRAY[@]}; i++))
do
    eval "${CMDARRAY[$i]}" &
    [ $(( i+1 % N_JOBS )) -eq 0 ]  && wait
done


