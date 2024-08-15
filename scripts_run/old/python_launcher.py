import subprocess
from multiprocessing import Pool

job_list = [
    # rnamigos1 no pretrain
    "python rnamigos/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16  model.decoder.out_dim=166 data.undirected=True train.target=native_fp train.use_rnamigos1_train=True name=retrain_lr4_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}",
    # add r_1 pretrain undirected
    "python rnamigos/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16  model.decoder.out_dim=166 data.undirected=True train.target=native_fp train.use_rnamigos1_train=True model.use_pretrained=True model.pretrained_path=pretrained/R_1_16_undirected/model.pth name=retrain_r1_undirected_lr4_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}",
    # rnamigos 1 architecture + new data
    "python rnamigos/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16 model.decoder.out_dim=166 data.undirected=True train.target=native_fp train.use_rnamigos1_train=False name=whole_data_lr4_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}",
    # with BCE loss
    "python rnamigos/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16 model.decoder.out_dim=166 data.undirected=True train.target=native_fp train.loss=bce name=bce_loss_lr4_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}",
    # add directed edges
    "python rnamigos/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16 model.decoder.out_dim=166 data.undirected=False train.target=native_fp train.loss=bce name=directed_lr4_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}",
    # add R1 pretrain
    "python rnamigos/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16 model.decoder.out_dim=166 train.target=native_fp train.loss=l2 model.use_pretrained=True model.pretrained_path=pretrained/R_1_16/model.pth name=pre_r1_l2_lr4_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}",
    # # R_graphlets
    # "python rnamigos/train.py model.encoder.hidden_dim=16 model.decoder.out_dim=166 train.target=native_fp loss=bce model.use_pretrained=True model.pretrained_path=pretrained/pretrained_model_Rgraphlets.pth name=pre_rgraphlets"
    # R_iso
    "python rnamigos/train.py model.encoder.hidden_dim=16 model.decoder.in_dim=16 model.decoder.out_dim=166 train.target=native_fp train.loss=l2 model.use_pretrained=True model.pretrained_path=pretrained/R_iso_16/model.pth name=pre_riso_small_l2_lr4_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}",
    # + Big model
    "python rnamigos/train.py model.encoder.hidden_dim=64 model.decoder.in_dim=64 model.decoder.out_dim=166 train.target=native_fp train.loss=l2 model.use_pretrained=True model.encoder.hidden_dim=64 model.pretrained_path=pretrained/R_iso_64/model.pth train.early_stop=50 name=pre_riso_big_lr4_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}",
    # add predict native training
    "python rnamigos/train.py model.encoder.hidden_dim=64 model.decoder.in_dim=96 model.decoder.out_dim=1 train.target=is_native train.loss=bce model.use_pretrained=True model.encoder.hidden_dim=64 model.pretrained_path=pretrained/R_iso_64/model.pth name=train_native_lr4_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID}",
    # docking score target
    "python rnamigos/train.py model.encoder.hidden_dim=64 model.decoder.out_dim=166 train.target=dock train.loss=l2 model.use_pretrained=True model.decoder.in_dim=96 model.decoder.out_dim=1 model.pretrained_path=pretrained/R_iso_64/model.pth name=train_dock_lr4_${SLURM_ARRAY_TASK_ID} train.rnamigos1_split=${SLURM_ARRAY_TASK_ID} model.decoder.activation=none train.num_workers=4"
]


def run_job(job):
    subprocess.run(job.split())


all_jobs = [job.replace("${SLURM_ARRAY_TASK_ID}", str(i)) + " train.learning_rate=1e-4"
            for i in range(10) for job in job_list]
pool = Pool(40)
pool.map(run_job, all_jobs)
