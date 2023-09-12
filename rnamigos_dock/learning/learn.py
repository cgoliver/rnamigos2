""" Trainer """

import sys
import time

import dgl
import torch
import torch.nn.functional as F

from rnamigos_dock.learning.utils import dgl_to_nx
from rnamigos_dock.learning.decoy_utils import *


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


def test(model, test_loader, criterion, device):
    """
    Compute accuracy and loss of model over given dataset
    :param model:
    :param test_loader:
    :param test_loss_fn:
    :param device:
    :return:
    """
    model.eval()
    test_loss = 0
    all_graphs = test_loader.dataset.dataset.all_graphs
    test_size = len(test_loader)
    for batch_idx, (graph, docked_fp, target, idx) in enumerate(test_loader):
        # Get data on the devices
        target = target.to(device)
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        with torch.no_grad():
            pred = model(graph, docked_fp)
            loss = criterion(target, pred)
        test_loss += loss.item()

    return test_loss / test_size


def train_dock(model,
               criterion,
               optimizer,
               train_loader,
               test_loader,
               save_path,
               writer=None,
               device='cpu',
               num_epochs=25,
               wall_time=None,
               early_stop_threshold=10,
               ):
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
    :param embed_only: number of epochs before starting attributor training.
    :return:
    """
    epochs_from_best = 0
    start_time = time.time()
    best_loss = sys.maxsize
    batch_size = train_loader.batch_size
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
        for batch_idx, (graph, docked_fp, target, idx) in enumerate(train_loader):
            # Get data on the devices
            # convert ints to one hots
            graph = send_graph_to_device(graph, device)
            target = target.to(device)
            pred = model(graph, docked_fp)

            loss = criterion(pred.squeeze(), target.float())

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

        # train_accuracy = running_corrects / num_batches
        # writer.log_scalar("Train accuracy during training", train_accuracy, epoch)

        # Test phase
        test_loss = test(model, test_loader, criterion, device)
        print(">> test loss ", test_loss)

        writer.add_scalar("Test loss during training", test_loss, epoch)

        ne = epoch + 1
        """
        learning_curve_val_df = learning_curve_val_df.append({'EPOCH': str(ne), 
            'LOSS': str(train_loss), 
            'TYPE_LOSS':'TRAIN'}, ignore_index=True)
        
        learning_curve_val_df = learning_curve_val_df.append({'EPOCH': str(ne), 
            'LOSS': str(test_loss), 
            'TYPE_LOSS':'TEST'}, ignore_index=True)
        """
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
    return best_loss


if __name__ == "__main__":
    pass
