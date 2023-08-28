import argparse
import os, sys
import pickle

import numpy as np

cwd = os.getcwd()
if cwd.endswith('learn'):
    sys.path.append('../')
else:
    sys.path.append('./')

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--parallel", default=True, help="If we don't want to run thing in parallel", action='store_false')
parser.add_argument("-da", "--annotated_data", default='pockets_nx_symmetric')
parser.add_argument("-bs", "--batch_size", type=int, default=8, help="choose the batch size")
parser.add_argument("-nw", "--workers", type=int, default=20, help="Number of workers to load data")
parser.add_argument("-n", "--name", type=str, default='default_name', help="Name for the logs")
parser.add_argument("-ep", "--num_epochs", type=int, help="number of epochs to train", default=3)
parser.add_argument("-fl", "--fp_lam", type=float, help="fingerprint lambda", default=1.0)
parser.add_argument("-rl", "--reconstruction_lam", type=float, help="reconstruction lambda", default=1.0)
parser.add_argument('-ad','--attributor_dims', nargs='+', type=int, help='Dimensions for attributor.', default=[16,166])
parser.add_argument('-ed','--embedding_dims', nargs='+', type=int, help='Dimensions for embeddings.', default=[16]*3)
parser.add_argument('-sf','--sim_function', type=str, help='Node similarity function (Supported Options: R_1, IDF).', default="R_1")
parser.add_argument('-eo','--embed_only', type=int, help='Number of epochs to train embedding network before starting attributor. If -eo > num_epochs, no attributions trained. If < 0, attributor and embeddings always on. If 0 <= -eo <- num_epochs, switch attributor ON and embeddings OFF.', default=-1)
parser.add_argument('-w', '--warm_start', type=str, default=None, help='Path to pre-trained model.')
parser.add_argument('-pw', '--pos_weight', type=int, default=0, help='Weight for positive examples.')
parser.add_argument('-po', '--pool', type=str, default='sum', help='Pooling function to use.')
parser.add_argument("-nu", "--nucs", default=True, help="Use nucleotide IDs for learn", action='store_false')
parser.add_argument('-rs', '--seed', type=int, default=0, help='Random seed to use (if > 0, else no seed is set).')
parser.add_argument('-kf', '--kfold', type=int, default=0, help='Do k-fold crossval and do decoys on each fold..')
parser.add_argument('-es', '--early_stop', type=int, default=10, help='Early stop epoch threshold (default=10)')
parser.add_argument('-cl', '--clustered',  action='store_true', default=False, help='Predict ligand cluster (default=False)')
parser.add_argument('-cn', '--num_clusts',  type=int, default=8, help='Number of clusters (default=8)')

args = parser.parse_args()

print("OPTIONS USED")
print("\n".join(map(str, zip(vars(args).items()))))
# Torch impors
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

if args.seed > 0:
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Homemade modules
import learning.learn as learn
from learning.loader import Loader
from learning.rgcn import Model

from learning.utils import mkdirs


print('Done importing')

'''
Hardware settings
'''

# torch.multiprocessing.set_sharing_strategy('file_system')
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# This is to create an appropriate number of workers, but works too with cpu
if args.parallel:
    used_gpus_count = torch.cuda.device_count()
else:
    used_gpus_count = 1

print(f'Using {used_gpus_count} GPUs')

'''
Dataloader creation
'''

annotated_file = 'data/annotated'
annotated_name = args.annotated_data
annotated_path = os.path.join(annotated_file, annotated_name)
print(f'Using {annotated_path} as the pocket inputs')

batch_size = args.batch_size
num_workers = args.workers

dims = args.embedding_dims
attributor_dims = args.attributor_dims

print(f'Using batch_size of {batch_size}')

loader = Loader(annotated_path=annotated_path,
                batch_size=batch_size, num_workers=num_workers,
                sim_function=args.sim_function,
                nucs=args.nucs)


    

print('Created data loader')

'''
Model loading
'''

#increase output embeddings by 1 for nuc info
if args.nucs:
    dim_add = 1
    attributor_dims[0] += 1
else:
    dim_add = 0

if dims[-1] != attributor_dims[0] - dim_add:
    raise ValueError(f"Final embedding size must match first attributor dimension: {dims[-1]} != {attributor_dims[0]}")

fp_lam = args.fp_lam
reconstruction_lam = args.reconstruction_lam

data = loader.get_data(k_fold=args.kfold)
for k, (train_loader, test_loader) in enumerate(data):
    model = Model(dims, device, attributor_dims=attributor_dims,
                  num_rels=loader.num_edge_types,
                  num_bases=-1, pool=args.pool,
                  pos_weight=args.pos_weight,
                  nucs=args.nucs, clustered=args.clustered,
                  num_clusts=args.num_clusts)

    #if pre-trained initialize matching layers
    if args.warm_start:
        print("warm starting")
        m = torch.load(args.warm_start, map_location='cpu')['model_state_dict']
        #remove keys not related to embeddings
        for key in list(m.keys()):
            if 'embedder' not in key:
                print("killing ", key)
                del m[key]
        missing = model.load_state_dict(m, strict=False)
        print(missing)

    model = model.to(device)

    print(f'Using {model.__class__} as model')

    '''
    Optimizer instanciation
    '''

    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    '''
    Experiment Setup
    '''

    name = f"{args.name}_{k}"
    print(name)
    result_folder, save_path = mkdirs(name)
    print(save_path)
    writer = SummaryWriter(result_folder)
    print(f'Saving result in {result_folder}/{name}')


    meta = {k:getattr(args, k) for k in dir(args) if not k.startswith("_")}
    meta['edge_map'] = train_loader.dataset.dataset.edge_map
    #save metainfo
    pickle.dump(meta, open(os.path.join(result_folder,  'meta.p'), 'wb'))


    all_graphs = np.array(test_loader.dataset.dataset.all_graphs)
    test_inds = test_loader.dataset.indices
    train_inds = train_loader.dataset.indices

    pickle.dump(({'test': all_graphs[test_inds], 'train': all_graphs[train_inds]}),
                    open(os.path.join(result_folder, f'splits_{k}.p'), 'wb'))

    '''
    Run
    '''
    num_epochs = args.num_epochs

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
                      fp_lam=fp_lam,
                      embed_only=args.embed_only,
                      early_stop_threshold=args.early_stop)
