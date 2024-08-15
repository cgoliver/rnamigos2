#!/bin/bash
#SBATCH --job-name="vs"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=12:00:00
#SBATCH --output vs_%a.out
#SBATCH --array=0-100%20

cd ..
source .venv/bin/activate

CMDARRAY=()
i=0

#for mode in native_fp is_native;
for mode in native_fp dock is_native;
do
	for model in `ls -d results/trained_models/${mode}/final*`;
	do
	model_name=$(basename $model)
	CMDARRAY+=("python rnamigos/evaluate.py saved_model_dir=results/trained_models/${mode}/${model_name}/ csv_name=${model_name}.csv")
	done
	for model in `ls -d results/trained_models/${mode}/definitive*`;
	do
	echo ${mode} ${model}
	model_name=$(basename $model)
	CMDARRAY+=("python rnamigos/evaluate.py saved_model_dir=results/trained_models/${mode}/${model_name}/ csv_name=${model_name}.csv")
	done
done

eval ${CMDARRAY[${SLURM_ARRAY_TASK_ID}]}

