#!/bin/bash

CMDARRAY=()
for stretch in 0 1
do
  # NATIVE
  python_cmd="python rnamigos/train.py
                    model.encoder.hidden_dim=64
                    model.use_pretrained=False
                    model.use_graphligs=True
                    model.graphlig_encoder.use_pretrained=True
                    model.decoder.in_dim=120
                    model.decoder.out_dim=1
                    model.decoder.activation=None
                    train.target=dock
                    train.loss=l2
                    train.use_normalized_score=True
                    train.stretch_scores=${stretch}
                    train.num_epochs=25
                    train.early_stop=20
                    train.num_workers=2
                    train.rnamigos1_split=-1
                    train.learning_rate=1e-3
                    train.pretrain_weight=0
                    train.simfunc=None
                    device=1
                    name=final_chembl_dock_graphligs_dim64_optimol1_quant_stretch${stretch}"
  python_cmd=$(echo $python_cmd) # to replace newlines
  CMDARRAY+=("$python_cmd")


  # NATIVE PRE W0
  python_cmd="python rnamigos/train.py
                    model.encoder.hidden_dim=64
                    model.use_pretrained=True
                    model.pretrained_path=pretrained/pretrain_hungarian_64/model.pth
                    model.use_graphligs=True
                    model.graphlig_encoder.use_pretrained=True
                    model.decoder.in_dim=120
                    model.decoder.out_dim=1
                    model.decoder.activation=None
                    train.target=dock
                    train.loss=l2
                    train.use_normalized_score=True
                    train.stretch_scores=${stretch}
                    train.num_epochs=25
                    train.early_stop=20
                    train.rnamigos1_split=-1
                    train.learning_rate=1e-3
                    train.num_workers=2
                    train.pretrain_weight=0
                    train.simfunc=None
                    device=2
                    name=final_chembl_dock_graphligs_dim64_simhungarian_prew0_optimol1_quant_stretch${stretch}"
  python_cmd=$(echo $python_cmd) # to replace newlines
  CMDARRAY+=("$python_cmd")
done

#echo "${CMDARRAY[0]}"

N_JOBS=10
for ((i = 0; i < ${#CMDARRAY[@]}; i++))
do
    eval "${CMDARRAY[$i]}" &
    [ $(( i+1 % N_JOBS )) -eq 0 ]  && wait
done




