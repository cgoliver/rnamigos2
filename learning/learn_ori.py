import time
import pickle
import torch
import torch.nn.functional as F
import sys
import pandas as pd

import dgl
import networkx as nx

#debug modules
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
##

if __name__ == '__main__':
    sys.path.append('../')

from learning.utils import dgl_to_nx
from learning.decoy_utils import *
from post.drawing import rna_draw

def send_graph_to_device(g, device):
    """
    Send dgl graph to device
    :param g: :param device:
    :return:
    """
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

    # edges
    labels = g.edge_attr_schemes()
    for i, l in enumerate(labels.keys()):
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)

    return g

def send_graph_to_device(g, device):
    """
    Send dgl graph to device
    :param g: :param device:
    :return:
    """
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

    # edges
    labels = g.edge_attr_schemes()
    for i, l in enumerate(labels.keys()):
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)

    return g
def set_gradients(model, embedding=True, attributor=True):
    """
        Set the gradients to the embedding and the attributor networks.
        If True sets requires_grad to true for network parameters.
    """
    for param in model.named_parameters():
        name, p = param
        name = name.split('.')[0]
        if name in ['attributor', 'pool']:
            p.requires_grad = attributor
        if name == 'embedder':
            p.requires_grad = embedding
    pass

def print_gradients(model):
    """
        Set the gradients to the embedding and the attributor networks.
        If True sets requires_grad to true for network parameters.
    """
    for param in model.named_parameters():
        name, p = param
        print(name, p.grad)
    pass
def test(model, test_loader, device, fp_lam=1, rec_lam=1):
    """
    Compute accuracy and loss of model over given dataset
    :param model:
    :param test_loader:
    :param test_loss_fn:
    :param device:
    :return:
    """
    model.eval()
    test_loss,  motif_loss_tot, recons_loss_tot = (0,) * 3
    all_graphs = test_loader.dataset.dataset.all_graphs
    test_size = len(test_loader)
    for batch_idx, (graph, K, fp, idx) in enumerate(test_loader):
        # Get data on the devices
        K = K.to(device)
        if model.clustered:
            fp = fp.long()

        fp = fp.to(device)
        K = torch.ones(K.shape).to(device) - K
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        with torch.no_grad():
            fp_pred, embeddings = model(graph)
            loss = model.compute_loss(fp, fp_pred, embeddings, K,
                                        fp_lam=fp_lam,
                                        rec_lam=rec_lam)
        del K
        del graph

        test_loss += loss.item()

        del loss
        del fp
        

    return test_loss / test_size

def train_model(model, criterion, optimizer, device, train_loader, test_loader, save_path,
                writer=None, num_epochs=25, wall_time=None,
                reconstruction_lam=1, fp_lam=1, embed_only=-1,
                early_stop_threshold=10, fp_draw=False):
    """
    Performs the entire training routine.
    :param model: (torch.nn.Module): the model to train
    :param criterion: the criterion to use (eg CrossEntropy)
    :param optimizer: the optimizer to use (eg SGD or Adam)
    :param device: the device on which to run
    :param train_loader: dataloader for training
    :param test_loader: dataloader for validation
    :param save_path: where to save the model
    :param writer: a Tensorboard object (defined in utils)
    :param num_epochs: int number of epochs
    :param wall_time: The number of hours you want the model to run
    :param reconstruction_lam: how much to enforce pariwise similarity conservation
    :param fp_lam: how much to enforce motif assignment
    :param embed_only: number of epochs before starting attributor training.
    :return:
    """

    edge_map = train_loader.dataset.dataset.edge_map
    all_graphs = train_loader.dataset.dataset.all_graphs

    epochs_from_best = 0

    start_time = time.time()
    best_loss = sys.maxsize

    fp_lam_orig = fp_lam
    reconstruction_lam_orig = reconstruction_lam

    batch_size = train_loader.batch_size
    #if we delay attributor, start with attributor OFF
    #if <= -1, both always ON.
    if embed_only > -1:
        print("Switching attriutor OFF. Embeddings still ON.")
        set_gradients(model, attributor=False)
        fp_lam = 0

    #learning_curve_val_df = pd.DataFrame(columns=['EPOCH', 'TRAINING_LOSS', 'TEST_LOSS'])
    learning_curve_val_df = pd.DataFrame(columns=['EPOCH', 'LOSS','TYPE_LOSS'])

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Training phase
        model.train()

        #switch off embedding grads, turn on attributor
        if epoch == embed_only:
            print("Switching attributor ON, embeddings OFF.")
            set_gradients(model, embedding=False, attributor=True)
            reconstruction_lam = 0
            fp_lam = fp_lam_orig

        running_loss = 0.0

        time_epoch = time.perf_counter()

        num_batches = len(train_loader)

        # learning_curve_val_df = pd.DataFrame(columns=['EPOCH', 'TRAINING_LOSS', 'TEST_LOSS'])

        for batch_idx, (graph, K, fp, idx) in enumerate(train_loader):

            # Get data on the devices
            #convert ints to one hots

            graph = send_graph_to_device(graph, device)
            if model.clustered:
                fp = fp.long()
            fp = fp.to(device)
            K = K.to(device)

            # for f, i in zip(fp, idx):
                # print(all_graphs[i.item()], f)

            fp_pred, embeddings = model(graph)

            loss = model.compute_loss(fp, fp_pred, embeddings, K,
                                      fp_lam=fp_lam, rec_lam=reconstruction_lam)

            # l = model.rec_loss(embeddings, K, similarity=False)
            # print(l)


            # Backward
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Metrics
            batch_loss = loss.item()
            running_loss += batch_loss

            if batch_idx % 20 == 0:
                time_elapsed = time.time() - start_time
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Time: {:.2f}'.format(
                    epoch + 1,
                    (batch_idx + 1) * batch_size,
                    num_batches * batch_size,
                    100. * (batch_idx + 1) / num_batches,
                    batch_loss,
                    time_elapsed))

                # tensorboard logging
                writer.add_scalar("Training batch loss", batch_loss,
                                  epoch * num_batches + batch_idx)

            del loss

        # Log training metrics
        train_loss = running_loss / num_batches
        writer.add_scalar("Training epoch loss", train_loss, epoch)
        print(">> Training epoch loss ", train_loss, epoch)

        # train_accuracy = running_corrects / num_batches
        # writer.log_scalar("Train accuracy during training", train_accuracy, epoch)

        # Test phase
        test_loss = test(model, test_loader, device, fp_lam=fp_lam, rec_lam=reconstruction_lam)
        print(">> test loss ", test_loss)

        writer.add_scalar("Test loss during training", test_loss, epoch)

        ne = epoch + 1 

        #learning_curve_val_df = learning_curve_val_df.append({'EPOCH': str(ne),
        #                                    'TRAINING_LOSS': str(train_loss),
        #                                    'TEST_LOSS': str(test_loss)}, ignore_index=True)
        learning_curve_val_df = learning_curve_val_df.append({'EPOCH': str(ne),
                                            'LOSS': str(train_loss),
                                            'TYPE_LOSS':'TRAIN'}, ignore_index=True)

        learning_curve_val_df = learning_curve_val_df.append({'EPOCH': str(ne),
                                            'LOSS': str(test_loss),
                                            'TYPE_LOSS':'TEST'}, ignore_index=True)


        # Checkpointing
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_from_best = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion
            }, save_path)

        # Early stopping
        else:
            epochs_from_best += 1
            if epochs_from_best > early_stop_threshold:
                print('This model was early stopped')
                break

        # Sanity Check
        if wall_time is not None:
            # Break out of the loop if we might go beyond the wall time
            time_elapsed = time.time() - start_time
            if time_elapsed * (1 + 1 / (epoch + 1)) > .95 * wall_time * 3600:
                break
        del test_loss

    print(learning_curve_val_df)

    # sns.lineplot(data=learning_curve_val_df, x="EPOCH", y="LOSS", hue='TYPE_LOSS')
    # plt.savefig(save_path + '/learning_curve_rnamigos_raw.png')
    print(save_path)
    pickle.dump(learning_curve_val_df, open('learning_curve_rnamigos_raw.p', 'wb'))

    return best_loss

if __name__ == "__main__":
    pass
