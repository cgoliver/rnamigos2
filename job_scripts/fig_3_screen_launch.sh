cd ..

#for mode in native_fp is_native;
for mode in native_fp dock is_native;
do
	for model in `ls -d results/trained_models/${mode}/definitive*`;
	do
	echo ${mode} ${model}
	model_name=$(basename $model)
	python experiments/evaluate.py saved_model_dir=results/trained_models/${mode}/${model_name}/ csv_name=${model_name}.csv
	done
	for model in `ls -d results/trained_models/${mode}/final*`;
	do
	model_name=$(basename $model)
	python experiments/evaluate.py saved_model_dir=results/trained_models/${mode}/${model_name}/ csv_name=${model_name}.csv
	done
done

