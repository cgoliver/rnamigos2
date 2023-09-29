#!/bin/bash

CMDARRAY=()

for split in {0..9};
do
  # R_1 undirected pre
  for prew in 0 1
  do
    python_cmd="python experiments/train.py
                          data.undirected=True
                          model.decoder.in_dim=16
                          model.decoder.out_dim=166
                          model.encoder.num_bases=null
                          model.encoder.hidden_dim=16
                          model.decoder.activation=sigmoid
                          model.use_pretrained=True
                          model.pretrained_path=pretrained/pretrain_R_1_undirected/model.pth
                          train.target=native_fp
                          train.loss=bce
                          train.num_epochs=1000
                          train.early_stop=100
                          train.learning_rate=1e-3
                          train.num_workers=0
                          train.rnamigos1_split=$split
                          train.use_rnamigos1_train=True
                          train.pretrain_weight=${prew}
                          train.simfunc=R_1
                          device=cpu
                          name=rnamigos1_repro_simR_1 prew_${prew}_$split"
    python_cmd=$(echo $python_cmd) # to replace newlines
    CMDARRAY+=("$python_cmd")
  done
done
#20xp
for split in {0..9};
do
  for simfunc in R_1 R_iso hungarian
  do
    for prew in 0 1
    do
      for dim in 16 64;
      do
      python_cmd="python experiments/train.py
                        model.encoder.hidden_dim=16
                        model.decoder.in_dim=${dim}
                        model.decoder.out_dim=166
                        model.encoder.hidden_dim=${dim}
                        model.decoder.activation=sigmoid
                        model.use_pretrained=True
                        model.pretrained_path=pretrained/pretrain_${simfunc}_${dim}/model.pth
                        train.target=native_fp
                        train.loss=bce
                        train.num_epochs=1000
                        train.early_stop=100
                        train.learning_rate=1e-3
                        train.num_workers=0
                        train.rnamigos1_split=${split}
                        train.use_rnamigos1_train=False
                        train.pretrain_weight=${prew}
                        train.simfunc=${simfunc}
                        device=cpu
                        name=rnamigos2_dim${dim}_sim${simfunc}_prew${prew}_${split}"
      python_cmd=$(echo $python_cmd) # to replace newlines
      CMDARRAY+=("$python_cmd")
      done
    done
  done
done
#120xp
N_JOBS=30
for ((i = 0; i < ${#CMDARRAY[@]}; i++))
do
    eval "${CMDARRAY[$i]}" &
    [ $(( i+1 % N_JOBS )) -eq 0 ]  && wait
done


