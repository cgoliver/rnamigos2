from dgl.dataloading import GraphDataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from rnamigos_dock.learning.loader import DockingDataset, get_systems
from rnamigos_dock.learning import learn
from rnamigos_dock.learning.models import Embedder, LigandEncoder, Decoder, RNAmigosModel
from rnamigos_dock.learning.utils import mkdirs


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print('Done importing')
    if cfg.train.seed > 0:
        torch.manual_seed(cfg.train.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    '''
    Hardware settings
    '''

    # torch.multiprocessing.set_sharing_strategy('file_system')
    if torch.cuda.is_available():
        device = 'cuda'
        # This is to create an appropriate number of workers, but works too with cpu
        if cfg.train.parallel:
            used_gpus_count = torch.cuda.device_count()
        else:
            used_gpus_count = 1
        print(f'Using {used_gpus_count} GPUs')
    else:
        device = 'cpu'
        print("No GPU found, running on the CPU")

    '''
    Dataloader creation
    '''

    use_rnamigos1_train = False
    use_rnamigos1_ligands = False
    rnamigos1_split = 0
    train_systems = get_systems(target=cfg.train.target,
                                rnamigos1_split=rnamigos1_split,
                                use_rnamigos1_train=use_rnamigos1_train,
                                use_rnamigos1_ligands=use_rnamigos1_ligands)
    test_systems = get_systems(target=cfg.train.target,
                               rnamigos1_split=rnamigos1_split,
                               use_rnamigos1_train=use_rnamigos1_train,
                               use_rnamigos1_ligands=use_rnamigos1_ligands,
                               return_test=True)
    dataset_args = {'pockets_path': cfg.data.pocket_graphs,
                    'target': cfg.train.target,
                    'shuffle': cfg.train.shuffle,
                    'edge_types': cfg.tokens.edge_types,
                    'seed': cfg.train.seed,
                    'debug': cfg.debug}
    loader_args = {'shuffle': True,
                   'batch_size': cfg.train.batch_size,
                   'num_workers': cfg.train.num_workers,
                   # 'collate_fn': None
                   }
    train_dataset = DockingDataset(systems=train_systems, **dataset_args)
    test_dataset = DockingDataset(systems=test_systems, **dataset_args)
    train_loader = GraphDataLoader(dataset=train_dataset, **loader_args)
    test_loader = GraphDataLoader(dataset=test_dataset, **loader_args)

    print('Created data loader')

    '''
    Model loading
    '''

    print("creating model")
    rna_encoder = Embedder(in_dim=cfg.model.encoder.in_dim,
                           hidden_dim=cfg.model.encoder.hidden_dim,
                           num_hidden_layers=cfg.model.encoder.num_layers,
                           )

    lig_encoder = LigandEncoder(in_dim=cfg.model.lig_encoder.in_dim,
                                hidden_dim=cfg.model.lig_encoder.hidden_dim,
                                num_hidden_layers=cfg.model.lig_encoder.num_layers)

    decoder = Decoder(in_dim=cfg.model.decoder.in_dim,
                      out_dim=cfg.model.decoder.out_dim,
                      hidden_dim=cfg.model.decoder.hidden_dim,
                      num_layers=cfg.model.decoder.num_layers)

    model = RNAmigosModel(encoder=rna_encoder,
                          decoder=decoder,
                          lig_encoder=lig_encoder if cfg.train.target in ['dock', 'is_native'] else None,
                          pool=cfg.model.pool,
                          pool_dim=cfg.model.encoder.hidden_dim
                          )

    if cfg.model.use_pretrained:
        model.from_pretrained(cfg.model.pretrained_path)

    model = model.to(device)

    print(f'Using {model.__class__} as model')

    '''
    Optimizer instanciation
    '''

    if cfg.train.loss == 'l2':
        criterion = torch.nn.MSELoss()
    if cfg.train.loss == 'l1':
        criterion = torch.nn.L1Loss()
    if cfg.train.loss == 'bce':
        criterion = torch.nn.BCELoss()

    optimizer = optim.Adam(model.parameters())

    '''
    Experiment Setup
    '''

    name = f"{cfg.name}"
    print(name)
    result_folder, save_path = mkdirs(name, prefix=cfg.train.target)
    print(save_path)
    writer = SummaryWriter(result_folder)
    print(f'Saving result in {result_folder}/{name}')

    '''
    Run
    '''
    num_epochs = cfg.train.num_epochs

    print("training...")
    learn.train_dock(model=model,
                     criterion=criterion,
                     optimizer=optimizer,
                     device=device,
                     train_loader=train_loader,
                     test_loader=test_loader,
                     save_path=save_path,
                     writer=writer,
                     num_epochs=num_epochs,
                     early_stop_threshold=cfg.train.early_stop)


if __name__ == "__main__":
    main()
