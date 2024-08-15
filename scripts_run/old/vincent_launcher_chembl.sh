#!/bin/bash

CMDARRAY=()

# FP no Pre
python_cmd="python rnamigos/train.py
                  model.encoder.hidden_dim=16
                  model.decoder.in_dim=64
                  model.decoder.out_dim=166
                  model.encoder.hidden_dim=64
                  model.decoder.activation=sigmoid
                  model.use_pretrained=False
                  train.target=native_fp
                  train.loss=bce
                  train.num_epochs=1000
                  train.early_stop=100
                  train.learning_rate=1e-3
                  train.num_workers=0
                  train.rnamigos1_split=-1
                  train.use_rnamigos1_train=False
                  train.pretrain_weight=0
                  train.simfunc=None
                  device=cpu
                  name=definitive_chembl_fp_dim64_simhungarian"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# FP PRE W0
python_cmd="python rnamigos/train.py
                  model.encoder.hidden_dim=16
                  model.decoder.in_dim=64
                  model.decoder.out_dim=166
                  model.encoder.hidden_dim=64
                  model.decoder.activation=sigmoid
                  model.use_pretrained=True
                  model.pretrained_path=pretrained/pretrain_hungarian_64/model.pth
                  train.target=native_fp
                  train.loss=bce
                  train.num_epochs=1000
                  train.early_stop=100
                  train.learning_rate=1e-3
                  train.num_workers=0
                  train.rnamigos1_split=-1
                  train.use_rnamigos1_train=False
                  train.pretrain_weight=0
                  train.simfunc=None
                  device=cpu
                  name=definitive_chembl_fp_dim64_simhungarian_prew0"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# FP PRE W1
python_cmd="python rnamigos/train.py
                  model.encoder.hidden_dim=16
                  model.decoder.in_dim=64
                  model.decoder.out_dim=166
                  model.encoder.hidden_dim=64
                  model.decoder.activation=sigmoid
                  model.use_pretrained=True
                  model.pretrained_path=pretrained/pretrain_hungarian_64/model.pth
                  train.target=native_fp
                  train.loss=bce
                  train.num_epochs=1000
                  train.early_stop=100
                  train.learning_rate=1e-3
                  train.num_workers=0
                  train.rnamigos1_split=-1
                  train.use_rnamigos1_train=False
                  train.pretrain_weight=1
                  train.simfunc=hungarian
                  device=cpu
                  name=definitive_chembl_fp_dim64_simhungarian_prew1"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# NATIVE
python_cmd="python rnamigos/train.py
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
                  name=definitive_chembl_native_dim64"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# NATIVE PRE W0
python_cmd="python rnamigos/train.py
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
                  name=definitive_chembl_native_dim64_simhungarian_prew0"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# NATIVE PRE W1
python_cmd="python rnamigos/train.py
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
                  name=definitive_chembl_native_dim64_simhungarian_prew1"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# DOCK
python_cmd="python rnamigos/train.py
                    model.encoder.hidden_dim=16
                    model.encoder.hidden_dim=64
                    model.decoder.in_dim=96
                    model.decoder.out_dim=1
                    model.decoder.activation=None
                    model.dropout=0.2
                    model.use_pretrained=False
                    train.target=dock
                    train.loss=l2
                    train.num_epochs=15
                    train.early_stop=10
                    train.learning_rate=1e-3
                    train.pretrain_weight=0
                    train.simfunc=None
                    train.rnamigos1_split=-1
                    name=definitive_chembl_dock_dim64
                    device=1
                    train.num_workers=2"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# DOCK PRE W0
python_cmd="python rnamigos/train.py
                    model.encoder.hidden_dim=16
                    model.encoder.hidden_dim=64
                    model.decoder.in_dim=96
                    model.decoder.out_dim=1
                    model.decoder.activation=None
                    model.dropout=0.2
                    model.use_pretrained=True
                    model.pretrained_path=pretrained/pretrain_hungarian_64/model.pth
                    train.target=dock
                    train.loss=l2
                    train.num_epochs=15
                    train.early_stop=10
                    train.learning_rate=1e-3
                    train.pretrain_weight=0
                    train.simfunc=None
                    train.rnamigos1_split=-1
                    name=definitive_chembl_dock_dim64_simhungarian_prew0
                    device=1
                    train.num_workers=2"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# DOCK PRE W1
python_cmd="python rnamigos/train.py
                    model.encoder.hidden_dim=16
                    model.encoder.hidden_dim=64
                    model.decoder.in_dim=96
                    model.decoder.out_dim=1
                    model.decoder.activation=None
                    model.dropout=0.2
                    model.use_pretrained=True
                    model.pretrained_path=pretrained/pretrain_hungarian_64/model.pth
                    train.target=dock
                    train.loss=l2
                    train.num_epochs=15
                    train.early_stop=10
                    train.learning_rate=1e-3
                    train.rnamigos1_split=-1
                    train.pretrain_weight=1
                    train.simfunc=hungarian
                    name=definitive_chembl_dock_dim64_simhungarian_prew1
                    device=2
                    train.num_workers=2"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

N_JOBS=10
for ((i = 0; i < ${#CMDARRAY[@]}; i++))
do
    eval "${CMDARRAY[$i]}" &
    [ $(( i+1 % N_JOBS )) -eq 0 ]  && wait
done

