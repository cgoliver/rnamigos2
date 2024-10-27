import sys

import numpy as np
import time
import torch

from rnamigos.utils.virtual_screen import run_virtual_screen
from rnamigos.utils.learning_utils import send_graph_to_device
from scripts_run.robin_inference import robin_eval


def print_gradients(model):
    """
    Set the gradients to the embedding and the attributor networks.
    If True sets requires_grad to true for network parameters.
    """
    for param in model.named_parameters():
        name, p = param
        print(name, p.grad.norm())
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
        graph, ligand_input, target, idx = (
            batch["graph"],
            batch["ligand_input"],
            batch["target"],
            batch["idx"],
        )

        # Get data on the devices
        ligand_input = ligand_input.to(device)
        target = target.to(device)
        graph = send_graph_to_device(graph, device)

        # Do the computations for the forward pass
        with torch.no_grad():
            pred, _ = model(graph, ligand_input)
            if criterion.__repr__() == "BCELoss()":
                loss = criterion(pred.squeeze(), target.squeeze(dim=0).float())
            else:
                loss = criterion(pred.squeeze(), target.float())
        val_loss += loss.item()

    return val_loss / val_size


def compute_rognan_loss(model, batch, mode="margin", alpha=0.3):
    """Only for positive pocket-ligand pairs (r,l) in the batch compute the loss as:
    $$ \max(0, p(r, l) - p(r', l) + \alpha) $$,
    where $\alpha$ is the margin and $p(.,.)$ is the model output for a pocket-ligand
    pair. We force decoy pockets $p'$ to have a lower score than the native pair.
    """
    pred_true = model(batch["graph"], batch["ligand_input"])[0].squeeze()
    pred_neg = model(batch["other_graph"], batch["ligand_input"])[0].squeeze()
    zero = torch.zeros_like(pred_true)
    y = batch["target"].detach()
    themax = torch.max(zero, pred_neg - pred_true + alpha)
    maxsum = torch.sum(y * themax)  # only keep true pocket-ligand pairs
    normed = (1 / torch.sum(y)) * maxsum
    return normed


def train_dock(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    val_vs_loader,
    test_vs_loader,
    save_path,
    val_vs_loader_rognan=None,
    writer=None,
    device="cpu",
    num_epochs=25,
    wall_time=None,
    early_stop_threshold=10,
    monitor_robin=False,
    pretrain_weight=0.1,
    negative_pocket="none",
    margin_only=False,
    debug=False,
    rognan_margin=0.3,
    cfg=None,
):
    """
    Performs the entire training routine.
    :param model: (torch.nn.Module): the model to train
    :param criterion: the criterion to use (e.g. CrossEntropy)
    :param optimizer: the optimizer to use (e.g. SGD or Adam)
    :param device: the device on which to run
    :param train_loader: dataloader for training
    :param val_loader: dataloader for validation
    :param save_path: where to save the model
    :param writer: a Tensorboard object (defined in utils)
    :param num_epochs: int number of epochs
    :param wall_time: The number of hours you want the model to run
    :return:
    """
    epochs_from_best = 0
    start_time = time.time()
    best_loss = sys.maxsize
    batch_size = train_loader.batch_size
    vs_every = cfg.train.vs_every if cfg is not None else 10
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)
        # Training phase
        model.train()
        running_loss = 0.0
        val_ef = 0
        for batch_idx, (batch) in enumerate(train_loader):
            # batch = test_batch
            graph, ligand_input, target, idx = (
                batch["graph"],
                batch["ligand_input"],
                batch["target"],
                batch["idx"],
            )

            node_sim_block, subsampled_nodes = batch["rings"]
            graph = send_graph_to_device(graph, device)
            ligand_input = ligand_input.to(device)
            target = target.to(device)
            pred, (embeddings, ligand_embeddings) = model(graph, ligand_input)
            if criterion.__repr__() == "BCELoss()":
                loss = criterion(pred.squeeze(), target.squeeze(dim=0).float())
            else:
                loss = criterion(pred.squeeze(), target.float())

            if negative_pocket != "none":
                if margin_only:
                    loss = 0
                rognan_loss = compute_rognan_loss(model, batch, alpha=rognan_margin)
                loss += rognan_loss
                pass

            # Optionally keep a small weight on the pretraining objective
            if pretrain_weight > 0 and node_sim_block is not None:
                subsampled_nodes_tensor = torch.tensor(subsampled_nodes, dtype=torch.bool)
                selected_embs = embeddings[torch.where(subsampled_nodes_tensor == 1)]
                node_sim_block = node_sim_block.to(device)
                K_predict = torch.mm(selected_embs, selected_embs.t())
                pretrain_loss = torch.nn.MSELoss()(K_predict, node_sim_block)
                loss += pretrain_weight * pretrain_loss

            # Backward
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.zero_grad()

            # Metrics
            batch_loss = loss.item()
            running_loss += batch_loss

            if batch_idx % 200 == 0:
                time_elapsed = time.time() - start_time
                print(
                    f"Train Epoch: {epoch + 1} [{(batch_idx + 1) * batch_size}/{num_batches * batch_size} "
                    f"({100. * (batch_idx + 1) / num_batches:.0f}%)]"
                    f"\tLoss: {batch_loss:.6f}  Time: {time_elapsed:.2f}"
                )

                # tensorboard logging
                writer.add_scalar("Training batch loss", batch_loss, epoch * num_batches + batch_idx)
                if negative_pocket != "none":
                    writer.add_scalar(
                        "Training non-pocket loss",
                        rognan_loss,
                        epoch * num_batches + batch_idx,
                    )
                    del rognan_loss
            del loss

        # Log training metrics
        train_loss = running_loss / num_batches
        writer.add_scalar("Training epoch loss", train_loss, epoch)

        # Validation phase
        val_loss = validate(model, val_loader, criterion, device)
        print(">> val loss ", val_loss)
        writer.add_scalar("Val loss", val_loss, epoch)

        # Run VS metrics
        if not epoch % vs_every and not debug:
            lower_is_better = cfg.train.target in ["dock", "native_fp"]
            efs, *_ = run_virtual_screen(model, val_vs_loader, lower_is_better=lower_is_better)
            val_ef = np.mean(efs)
            writer.add_scalar("Val EF", val_ef, epoch)

            if val_vs_loader_rognan is not None:
                efs, *_ = run_virtual_screen(model, val_vs_loader_rognan, lower_is_better=lower_is_better)
                writer.add_scalar("Val EF Rognan", np.mean(efs), epoch)

            efs, *_ = run_virtual_screen(model, test_vs_loader, lower_is_better=lower_is_better)
            writer.add_scalar("Test EF", np.mean(efs), epoch)

            if monitor_robin:
                robin_ef05 = robin_eval(cfg, model)
                writer.add_scalar("Robin EF", robin_ef05, epoch)

        # Finally do checkpointing based on vs performance, we negate it since higher efs are better
        # loss_to_track = val_loss
        loss_to_track = -val_ef

        # Checkpointing
        if loss_to_track < best_loss:
            best_loss = loss_to_track
            epochs_from_best = 0
            model.cpu()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                save_path,
            )
            model.to(device)

        # Early stopping
        else:
            epochs_from_best += 1
            if epochs_from_best > early_stop_threshold:
                print("This model was early stopped")
                break

        # Sanity Check
        if wall_time is not None:
            # Break out of the loop if we might go beyond the wall time
            time_elapsed = time.time() - start_time
            if time_elapsed * (1 + 1 / (epoch + 1)) > 0.95 * wall_time * 3600:
                break
        del val_loss
    best_state_dict = torch.load(save_path)["model_state_dict"]
    model.load_state_dict(best_state_dict)
    model.eval()
    return best_loss, model


if __name__ == "__main__":
    pass
