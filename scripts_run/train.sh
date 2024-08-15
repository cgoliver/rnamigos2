# Historically, these models were named native_4 and dock_2 after a few tries,
# we rename them for publication

CMDARRAY=()

for seed in 0 1 42;
  do
  # NATIVE MODEL
  python_cmd="python rnamigos/train.py
              model.use_pretrained=True
              model.pretrained_path=pretrained/pretrain_hungarian_64/model.pth
              model.use_graphligs=True
              model.graphlig_encoder.use_pretrained=True
              model.decoder.in_dim=120
              model.decoder.out_dim=1
              model.decoder.activation=sigmoid
              train.target=is_native
              train.loss=bce
              train.num_epochs=1000
              train.early_stop=100
              train.rnamigos1_split=-2
              train.group_pockets=1
              device=cpu
              train.learning_rate=0.0001
              name=native_${seed}"
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
              train.num_epochs=40
              train.early_stop=30
              train.rnamigos1_split=-2
              train.num_workers=2
              train.group_pockets=1
              device=cpu
              train.vs_every=2
              name=dock_${seed}"
  python_cmd=$(echo $python_cmd) # to replace newlines
  CMDARRAY+=("$python_cmd")
done

for ((i = 0; i < ${#CMDARRAY[@]}; i++))
do
#    echo "${CMDARRAY[$i]}" &
    eval "${CMDARRAY[$i]}" &
done