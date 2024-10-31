CMDARRAY=()

for seed in 0 1 42;
  do
  # NATIVE MODEL
  python_cmd="python rnamigos/train.py
              train.target=is_native
              train.loss=bce
              train.num_epochs=3000
              train.early_stop=300
              train.negative_pocket=rognan
              train.validation_systems=true
              train.group_sample=True
              train.bce_weight=0.02
              train.monitor_gap=True
              model.dropout=0.5
              model.decoder.dropout=0.5
              model.decoder.bn_all_layers=False
              name=native_${seed}"
  python_cmd=$(echo $python_cmd) # to replace newlines
  CMDARRAY+=("$python_cmd")


  # DOCK MODEL
  python_cmd="python rnamigos/train.py
               train.target=dock
               train.loss=l2
               train.num_epochs=80
               train.early_stop=30
               train.num_workers=2
               train.vs_every=2
               model.decoder.activation=None
               model.dropout=0.2
               model.decoder.dropout=0.2
                name=dock_${seed}"
  python_cmd=$(echo $python_cmd) # to replace newlines
  CMDARRAY+=("$python_cmd")
done

for ((i = 0; i < ${#CMDARRAY[@]}; i++))
do
#    echo "${CMDARRAY[$i]}" &
    eval "${CMDARRAY[$i]}" &
done