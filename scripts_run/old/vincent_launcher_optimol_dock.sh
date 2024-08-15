#!/bin/bash

CMDARRAY=()
for pre in 0 1
do
  # NATIVE
  python_cmd="python rnamigos/train.py
                    model.encoder.hidden_dim=64
                    model.decoder.in_dim=$((pre*56+(1-pre)*32+64))
                    model.decoder.out_dim=1
                    model.decoder.activation=None
                    model.use_pretrained=False
                    model.use_graphligs=True
                    model.graphlig_encoder.use_pretrained=${pre}
                    train.target=dock
                    train.loss=l2
                    train.num_epochs=40
                    train.early_stop=20
                    device=0
                    train.num_workers=2
                    train.rnamigos1_split=-1
                    train.early_stop=100
                    train.learning_rate=1e-3
                    train.pretrain_weight=0
                    train.simfunc=None
                    name=final_chembl_dock_graphligs_dim64_optimol${pre}"
  python_cmd=$(echo $python_cmd) # to replace newlines
  CMDARRAY+=("$python_cmd")


  # NATIVE PRE W0
  python_cmd="python rnamigos/train.py
                    model.encoder.hidden_dim=64
                    model.decoder.in_dim=$((pre*56+(1-pre)*32+64))
                    model.decoder.out_dim=1
                    model.decoder.activation=None
                    model.use_pretrained=True
                    model.pretrained_path=pretrained/pretrain_hungarian_64/model.pth
                    model.use_graphligs=True
                    model.graphlig_encoder.use_pretrained=${pre}
                    train.target=dock
                    train.loss=l2
                    train.num_epochs=40
                    train.rnamigos1_split=-1
                    train.early_stop=20
                    train.learning_rate=1e-3
                    device=2
                    train.num_workers=2
                    train.pretrain_weight=0
                    train.simfunc=None
                    name=final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol${pre}"
  python_cmd=$(echo $python_cmd) # to replace newlines
  CMDARRAY+=("$python_cmd")


  # NATIVE PRE W1
  python_cmd="python rnamigos/train.py
                    model.encoder.hidden_dim=64
                    model.decoder.in_dim=$((pre*56+(1-pre)*32+64))
                    model.decoder.out_dim=1
                    model.decoder.activation=None
                    model.use_pretrained=True
                    model.pretrained_path=pretrained/pretrain_hungarian_64/model.pth
                    model.use_graphligs=True
                    model.graphlig_encoder.use_pretrained=${pre}
                    train.target=dock
                    train.loss=l2
                    train.num_epochs=40
                    train.early_stop=20
                    train.rnamigos1_split=-1
                    train.learning_rate=1e-3
                    device=2
                    train.num_workers=2
                    train.pretrain_weight=1
                    train.simfunc=hungarian
                    name=final_chembl_dock_graphligs_dim64_simhungarian_prew1_optimol${pre}"
  python_cmd=$(echo $python_cmd) # to replace newlines
  CMDARRAY+=("$python_cmd")
done

N_JOBS=10
for ((i = 0; i < ${#CMDARRAY[@]}; i++))
do
    eval "${CMDARRAY[$i]}" &
    [ $(( i+1 % N_JOBS )) -eq 0 ]  && wait
done




