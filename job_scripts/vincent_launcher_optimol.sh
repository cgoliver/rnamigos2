#!/bin/bash

CMDARRAY=()

# NATIVE
python_cmd="python experiments/train.py
                  model.encoder.hidden_dim=16
                  model.encoder.hidden_dim=64
                  model.decoder.in_dim=96
                  model.decoder.out_dim=1
                  model.decoder.activation=sigmoid
                  model.use_pretrained=False
                  train.target=is_native
                  train.loss=bce
                  train.num_epochs=1000
                  train.rnamigos1_split=-1
                  train.early_stop=100
                  train.learning_rate=1e-3
                  device=cpu
                  train.num_workers=0
                  train.pretrain_weight=0
                  train.simfunc=None
                  name=final_chembl_native_dim64"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# NATIVE PRE W0
python_cmd="python experiments/train.py
                  model.encoder.hidden_dim=16
                  model.encoder.hidden_dim=64
                  model.decoder.in_dim=96
                  model.decoder.out_dim=1
                  model.decoder.activation=sigmoid
                  model.use_pretrained=True
                  model.pretrained_path=pretrained/pretrain_hungarian_64/model.pth
                  train.target=is_native
                  train.loss=bce
                  train.num_epochs=1000
                  train.rnamigos1_split=-1
                  train.early_stop=100
                  train.learning_rate=1e-3
                  device=cpu
                  train.num_workers=0
                  train.pretrain_weight=0
                  train.simfunc=None
                  name=final_chembl_native_dim64_simhungarian_prew0"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# NATIVE PRE W1
python_cmd="python experiments/train.py
                  model.encoder.hidden_dim=16
                  model.encoder.hidden_dim=64
                  model.decoder.in_dim=96
                  model.decoder.out_dim=1
                  model.decoder.activation=sigmoid
                  model.use_pretrained=True
                  model.pretrained_path=pretrained/pretrain_hungarian_64/model.pth
                  train.target=is_native
                  train.loss=bce
                  train.num_epochs=1000
                  train.rnamigos1_split=-1
                  train.early_stop=100
                  train.learning_rate=1e-3
                  device=cpu
                  train.num_workers=0
                  train.pretrain_weight=1
                  train.simfunc=hungarian
                  name=final_chembl_native_dim64_simhungarian_prew1"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

N_JOBS=10
for ((i = 0; i < ${#CMDARRAY[@]}; i++))
do
    eval "${CMDARRAY[$i]}" &
    [ $(( i+1 % N_JOBS )) -eq 0 ]  && wait
done

