import argparse
import os, sys
import pickle
import copy
import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
import hydra

import learning.learn as learn
from learning.loader import Loader
from learning.rgcn import Model
from learning.utils import mkdirs


@hydra.main(version_base=None, config_path="../conf", config_name="learning")
def main(cfg: DictConfig):

    print('Done importing')
    if cfg.train.seed > 0:
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    '''
    Hardware settings
    '''

    # torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    # This is to create an appropriate number of workers, but works too with cpu
    if cfg.train.parallel:
        used_gpus_count = torch.cuda.device_count()
    else:
        used_gpus_count = 1

    print(f'Using {used_gpus_count} GPUs')

    '''
    Dataloader creation
    '''


    loader = Loader(annotated_path=cfg.data.train_graphs,
                    batch_size=cfg.train.batch_size, 
                    num_workers=cfg.train.num_workers,
                    sim_function=cfg.pretrain.sim_function,
                    nucs=cfg.data.use_nucs)

    print('Created data loader')

    '''
    Model loading
    '''

    #increase output embeddings by 1 for nuc info
    clf_lam = cfg.train.clf_lam
    reconstruction_lam = cfg.train.reconstruction_lam

    data = loader.get_data(k_fold=cfg.train.kfold)
    attributor_dims_init = copy.deepcopy(attributor_dims)

    for k, (train_loader, test_loader) in enumerate(data):
        model = Model(dims, 
                      device, 
                      attributor_dims=attributor_dims,
                      num_rels=loader.num_edge_types,
                      num_bases=-1,
                      pool=args.pool,
                      pos_weight=args.pos_weight,
                      nucs=args.nucs
                      )

        #if pre-trained initialize matching layers
        if cfg.train.warm_start:
            print("warm starting")
            m = torch.load(args.warm_start, map_location='cpu')
            #remove keys not related to embeddings
            for key in list(m.keys()):
                if 'embedder' not in key:
                    print("killing ", key)
                    del m[key]
            missing = model.load_state_dict(m, strict=False)

        model = model.to(device)

        print(f'Using {model.__class__} as model')

        '''
        Optimizer instanciation
        '''

        criterion = torch.nn.BCELoss()
        #criterion = torch.nn.L1Loss()
        optimizer = optim.Adam(model.parameters())

        '''
        Experiment Setup
        '''
        
        name = f"{cfg.name}_{k}"
        print(name)
        result_folder, save_path = mkdirs(name)
        print(save_path)
        writer = SummaryWriter(result_folder)
        print(f'Saving result in {result_folder}/{name}')


        
        all_graphs = np.array(test_loader.dataset.dataset.all_graphs)
        test_inds = test_loader.dataset.indices
        train_inds = train_loader.dataset.indices

        pickle.dump(({'test': all_graphs[test_inds], 'train': all_graphs[train_inds]}),
                        open(os.path.join(result_folder, f'splits_{k}.p'), 'wb'))

        '''
        Run
        '''
        num_epochs = cfg.train.num_epochs

        learn.train_model(model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          device=device,
                          train_loader=train_loader,
                          test_loader=test_loader,
                          save_path=save_path,
                          writer=writer,
                          num_epochs=num_epochs,
                          reconstruction_lam=reconstruction_lam,
                          clf_lam=clf_lam,
                          embed_only=cfg.train.embed_only,
                          early_stop_threshold=cfg.train.early_stop)
        
if __name__ == "__main__":
    main()
