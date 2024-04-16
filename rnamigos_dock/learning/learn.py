""" Trainer """

import sys

import dgl
import time
import torch

from rnamigos_dock.learning.decoy_utils import *
from rnamigos_dock.post.virtual_screen import mean_active_rank, run_virtual_screen


def send_graph_to_device(g, device):
    """
    Send dgl graph to device
    :param g: :param device:
    :return:
    """
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    g = g.to(device)
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device, non_blocking=True)

    # edges
    labels = g.edge_attr_schemes()
    for i, l in enumerate(labels.keys()):
        g.edata[l] = g.edata.pop(l).to(device, non_blocking=True)
    return g


def print_gradients(model):
    """
        Set the gradients to the embedding and the attributor networks.
        If True sets requires_grad to true for network parameters.
    """
    for param in model.named_parameters():
        name, p = param
        print(name, p.grad)
    pass


def validate(model, val_loader, criterion, device):
    """
    Compute accuracy and loss of model over given dataset
    :param model:
    :param val_loader:
    :param criterion:
    :param device:
    :return:
    """
    model.eval()
    val_loss = 0
    val_size = len(val_loader)
    for batch_idx, (batch) in enumerate(val_loader):
        graph, ligand_input, target, idx = batch['graph'], batch['ligand_input'], batch['target'], batch['idx']

        # Get data on the devices
        ligand_input = ligand_input.to(device)
        target = target.to(device)
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        with torch.no_grad():
            pred, _ = model(graph, ligand_input)
            if criterion.__repr__() == 'BCELoss()':
                loss = criterion(pred.squeeze(), target.squeeze(dim=0).float())
            else:
                loss = criterion(pred.squeeze(), target.float())
        val_loss += loss.item()

    return val_loss / val_size


def train_dock(model,
               criterion,
               optimizer,
               train_loader,
               val_loader,
               val_vs_loader,
               test_vs_loader,
               save_path,
               writer=None,
               device='cpu',
               num_epochs=25,
               wall_time=None,
               early_stop_threshold=10,
               pretrain_weight=0.1,
               cfg=None
               ):
    """
    Performs the entire training routine.
    :param model: (torch.nn.Module): the model to train
    :param criterion: the criterion to use (eg CrossEntropy)
    :param optimizer: the optimizer to use (eg SGD or Adam)
    :param device: the device on which to run
    :param train_loader: dataloader for training
    :param val_loader: dataloader for validation
    :param save_path: where to save the model
    :param writer: a Tensorboard object (defined in utils)
    :param num_epochs: int number of epochs
    :param wall_time: The number of hours you want the model to run
    :param embed_only: number of epochs before starting attributor training.
    :return:
    """
    epochs_from_best = 0
    start_time = time.time()
    best_loss = sys.maxsize
    batch_size = train_loader.batch_size
    vs_every = cfg.train.vs_every if cfg is not None else 10
    # if we delay attributor, start with attributor OFF
    # if <= -1, both always ON.

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Training phase
        model.train()

        # switch off embedding grads, turn on attributor
        running_loss = 0.0
        time_epoch = time.perf_counter()
        num_batches = len(train_loader)
        for batch_idx, (batch) in enumerate(train_loader):
            graph, ligand_input, target, idx = batch['graph'], batch['ligand_input'], batch['target'], batch['idx']
            node_sim_block, subsampled_nodes = batch['rings']
            # Get data on the devices
            # convert ints to one hots
            graph = send_graph_to_device(graph, device)
            ligand_input = ligand_input.to(device)
            target = target.to(device)
            pred, embeddings = model(graph, ligand_input)
            if criterion.__repr__() == 'BCELoss()':
                loss = criterion(pred.squeeze(), target.squeeze(dim=0).float())
            else:
                loss = criterion(pred.squeeze(), target.float())

            if pretrain_weight > 0 and node_sim_block is not None:
                subsampled_nodes_tensor = torch.tensor(subsampled_nodes, dtype=torch.bool)
                selected_embs = embeddings[torch.where(subsampled_nodes_tensor == 1)]
                node_sim_block = node_sim_block.to(device)
                K_predict = torch.mm(selected_embs, selected_embs.t())
                pretrain_loss = torch.nn.MSELoss()(K_predict, node_sim_block)
                loss += pretrain_weight * pretrain_loss
            # Backward
            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Metrics
            batch_loss = loss.item()
            running_loss += batch_loss

            if batch_idx % 200 == 0:
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

        # Validation phase
        val_loss = validate(model, val_loader, criterion, device)
        print(">> val loss ", val_loss)

        writer.add_scalar("Validation loss during training", val_loss, epoch)

        """
        learning_curve_val_df = learning_curve_val_df.append({'EPOCH': str(ne), 
            'LOSS': str(train_loss), 
            'TYPE_LOSS':'TRAIN'}, ignore_index=True)
        
        learning_curve_val_df = learning_curve_val_df.append({'EPOCH': str(ne), 
            'LOSS': str(val_loss), 
            'TYPE_LOSS':'VAL'}, ignore_index=True)
        """
        # Checkpointing
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_from_best = 0
            model.cpu()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion
            }, save_path)
            model.to(device)

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
        del val_loss

        if not epoch % vs_every:
            lower_is_better = cfg.train.target in ['dock', 'native_fp']

            efs, scores, status, pocket_names, all_smiles = run_virtual_screen(model,
                                                                               val_vs_loader,
                                                                               metric=mean_active_rank,
                                                                               lower_is_better=lower_is_better)
            writer.add_scalar("Val EF during training", np.mean(efs), epoch)

            efs, scores, status, pocket_names, all_smiles = run_virtual_screen(model,
                                                                               test_vs_loader,
                                                                               metric=mean_active_rank,
                                                                               lower_is_better=lower_is_better)
            writer.add_scalar("Test EF during training", np.mean(efs), epoch)

    best_state_dict = torch.load(save_path)['model_state_dict']
    model.load_state_dict(best_state_dict)
    model.eval()
    return best_loss, model


if __name__ == "__main__":
    pass
