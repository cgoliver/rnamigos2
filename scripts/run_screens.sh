#!/bin/bash
#SBATCH --job-name="inference"
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-core=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=12:00:00
#SBATCH --output inf_%a.out
#SBATCH --array=0-100%20

cd ..
source .venv/bin/activate

CMDARRAY=()

for target in dock is_native native_fp;
do
	for run in `ls results/trained_models/${target}`;
	do
		echo results/trained_models/${target}/${run}
		CMDARRAY+=("python experiments/evaluate.py saved_model_dir=results/trained_models/${target}/${run}  csv_name=${run}.csv")
	done
done		

eval ${CMDARRAY[${SLURM_ARRAY_TASK_ID}]}
		
