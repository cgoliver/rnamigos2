import time
import torch
import sys
import dgl

if __name__ == '__main__':
    sys.path.append('../')


def send_graph_to_device(g, device):
    """
    Send dgl graph to device
    :param g:
    :param device:
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


def test(model, test_loader, test_loss_fn, device):
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
    test_size = len(test_loader)
    for batch_idx, (graph, K) in enumerate(test_loader):
        # Get data on the devices
        K = K.to(device)
        K = torch.ones(K.shape).to(device) - K
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        out = model(graph)
        K_predict = torch.norm(out[:, None] - out, dim=2, p=2)

        loss = test_loss_fn(K_predict, K)
        test_loss += loss.item()

    return test_loss / test_size

def train_model(model, criterion, optimizer, device, train_loader, test_loader, save_path,
                writer=None, num_epochs=25, wall_time=None):
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
    :return:
    """

    # print('forward', torch.cuda.memory_allocated(device=device))
    # print(torch.cuda.memory_cached(device=device))
    # print('enter loop', torch.cuda.memory_allocated(device=device))
    # print(torch.cuda.memory_cached(device=device))

    # torch.cuda.synchronize()  # wait for mm to finish

    epochs_from_best = 0
    early_stop_threshold = 60

    start_time = time.time()
    best_loss = sys.maxsize

    tloop = 0
    step = 5
    for batch_idx, (graph, K) in enumerate(train_loader):
        # Get data on the devices
        if not batch_idx % step:
            print(f'loop in : {(time.perf_counter() - tloop) / step}')
            tloop = time.perf_counter()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Training phase
        model.train()

        running_loss = 0.0

        total_loss = 0
        time_epoch = time.perf_counter()

        num_batches = len(train_loader)
        tloop = time.perf_counter()

        for batch_idx, (graph, K) in enumerate(train_loader):
            # Get data on the devices
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f'loop in : {time.perf_counter() - tloop}')
            tloop = time.perf_counter()

            tdata = time.perf_counter()
            batch_size = len(K)
            K = K.to(device)
            K = torch.ones(K.shape).to(device) - K
            graph = send_graph_to_device(graph, device)
            print(f'data in : {time.perf_counter() - tdata}')
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Do the computations for the forward pass

            tfor = time.perf_counter()
            optimizer.zero_grad()
            out = model(graph)
            K_predict = torch.norm(out[:, None] - out, dim=2, p=2)
            print(f'loop in : {time.perf_counter() - tfor}')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            # Backward

            tback = time.perf_counter()

            loss = criterion(K_predict, K)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(f'back in : {time.perf_counter() - tback}')
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f'computation total in : {time.perf_counter() - tloop}')
            print()

            # Metrics
            batch_loss = loss.item()
            del loss
            running_loss += batch_loss

            # running_corrects += labels.eq(target.view_as(out)).sum().item()
            if batch_idx % 20 == 0:
                time_elapsed = time.time() - start_time
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Time: {:.2f}'.format(
                    epoch + 1,
                    (batch_idx + 1) * batch_size,
                    num_batches * batch_size,
                    100. * (batch_idx + 1) / num_batches,
                    batch_loss,
                    time_elapsed))

                continue
                # tensorboard logging
                writer.log_scalar("Training batch loss", batch_loss,
                                  epoch * num_batches + batch_idx)

        continue
        # Log training metrics
        train_loss = running_loss / num_batches
        writer.log_scalar("Training epoch loss", train_loss, epoch)

        # train_accuracy = running_corrects / num_batches
        # writer.log_scalar("Train accuracy during training", train_accuracy, epoch)

        # Test phase
        test_loss, test_accuracy = test(model, test_loader, criterion, device)
        writer.log_scalar("Test loss during training", test_loss, epoch)
        # writer.log_scalar("Test accuracy during training", test_accuracy, epoch)

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
    return best_loss
