#!/bin/bash
#
cd ..
source .venv/bin/activate


for target in dock is_native native_fp;
do
	for run in `ls results/trained_models/${target}`;
	do
		echo results/trained_models/${target}/${run}
		python experiments/evaluate.py saved_model_dir=results/trained_models/${target}/${run} \
					       csv_name=${run}.csv
	done
done		
		
