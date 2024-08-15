CMDARRAY=()

# FP MODEL
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
                  train.filter_robin=True
                  device=cpu
                  name=robin_fp"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# NATIVE MODEL
python_cmd="python rnamigos/train.py
                    model.encoder.hidden_dim=64
                    model.decoder.in_dim=120
                    model.decoder.out_dim=1
                    model.decoder.activation=sigmoid
                    model.use_graphligs=True
                    model.graphlig_encoder.use_pretrained=True
                    train.target=is_native
                    train.loss=bce
                    train.num_epochs=1000
                    train.early_stop=100
                    train.filter_robin=True
                    device=cpu
                    name=robin_native"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

# DOCK MODEL
python_cmd="python rnamigos/train.py
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
                    train.num_epochs=25
                    train.early_stop=20
                    train.num_workers=2
                    train.filter_robin=True
                    device=0
                    name=robin_dock"
python_cmd=$(echo $python_cmd) # to replace newlines
CMDARRAY+=("$python_cmd")

for ((i = 0; i < ${#CMDARRAY[@]}; i++))
do
    eval "${CMDARRAY[$i]}" &
done