cd ..
#python rnamigos/pretrain.py name=R_1_16_undirected simfunc=R_1 model.encoder.hidden_dim=16 epochs=20 data.undirected=True
python rnamigos/pretrain.py name=R_graphlets_16 simfunc=R_graphlets model.encoder.hidden_dim=16 epochs=20;

#for size in 16 64;
#do
#    for simfunc in R_1 R_iso R_graphlets hungarian;
#    do
#        echo ${size} ${simfunc};
#        python rnamigos/pretrain.py name=${simfunc}_${size} simfunc=${simfunc} model.encoder.hidden_dim=${size} epochs=20;
#    done
#done

