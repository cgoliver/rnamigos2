import sys
import random

import numpy as np
import time
import torch
import dgl

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


def double_decoy_outputs(model, batch):
    # score on rognan pockets, only compute within PDB ligands
    pockets = dgl.unbatch(batch["graph"])
    ligands = dgl.unbatch(batch["ligand_input"])
    y = batch["target"]

    pos_ligands = dgl.batch([l for i, l in enumerate(ligands) if y[i] == 1])
    random_ligs = dgl.batch(random.sample(ligands, torch.sum(y)))

    pos_pockets = [p for i, p in enumerate(pockets) if y[i] == 1]
    pos_pockets_shuffle = random.sample(pos_pockets, len(pos_pockets))
    pos_pockets_shuffle = dgl.batch(pos_pockets_shuffle)
    pos_pockets = dgl.batch(pos_pockets)

    pos_scores = model(pos_pockets, pos_ligands)[0]  # true native pairs
    scores_p_prime = model(pos_pockets_shuffle, pos_ligands)[0]  # shuffled pockets vs PDB ligands
    scores_l_prime = model(pos_pockets_shuffle, random_ligs)[0]  # PDB pockets vs pdbchembl ligands

    return pos_scores, scores_p_prime, scores_l_prime


def compute_rognan_loss(model, batch, mode="rognan", loss="margin", alpha=0.3):
    """Only for positive pocket-ligand pairs (r,l) in the batch compute the loss as:
    $$ \max(0, p(r, l) - p(r', l) + \alpha) $$,
    where $\alpha$ is the margin and $p(.,.)$ is the model output for a pocket-ligand
    pair. We force decoy pockets $p'$ to have a lower score than the native pair.
    """

    def get_margin(true, neg, weights=None):
        zero = torch.zeros_like(true)
        if weights is None:
            weights = torch.ones_like(true)
        return torch.sum(weights * torch.max(zero, neg - true + alpha))

    def get_exp(true, neg):
        return torch.exp(-1 * true / (neg + 0.0000001))

    y = batch["target"].detach()
    n_pos = torch.sum(y)

    if mode == "rognan":
        pred_true = model(batch["graph"], batch["ligand_input"])[0].squeeze()
        pred_neg = model(batch["other_graph"], batch["ligand_input"])[0].squeeze()
        if loss == "margin":
            margin = get_margin(pred_true, pred_neg, weights=y)
            tot = (1 / n_pos) * margin

    if mode == "double":
        pos_pred, p_prime, l_prime = double_decoy_outputs(model, batch)
        if loss == "margin":
            tot = (1 / pos_pred.size(0)) * get_margin(pos_pred, (1 / 2) * (p_prime + l_prime))
        if loss == "exp":
            tot = get_exp(pos_pred.mean(), torch.mean(p_prime + l_prime))

    return tot


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
    bce_weight=1.0,
    debug=False,
    rognan_margin=0.3,
    rognan_lossfunc="margin",
    cfg=None,
):
    """
    Performs the entire training routine
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

            loss = bce_weight * loss
            if negative_pocket != "none":
                rognan_loss = compute_rognan_loss(
                    model, batch, alpha=rognan_margin, mode=negative_pocket, loss=rognan_lossfunc
                )
                loss += rognan_loss

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
                    writer.add_scalar("Training non-pocket loss", rognan_loss, epoch * num_batches + batch_idx)
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
            val_efs, *_ = run_virtual_screen(model, val_vs_loader, lower_is_better=lower_is_better)
            val_ef = np.mean(val_efs)
            writer.add_scalar("Val EF", val_ef, epoch)

            if val_vs_loader_rognan is not None:
                rognan_efs, *_ = run_virtual_screen(model, val_vs_loader_rognan, lower_is_better=lower_is_better)
                writer.add_scalar("Val EF Rognan", np.mean(rognan_efs), epoch)
                gap_val = 2 * val_ef - np.mean(rognan_efs)
                writer.add_scalar("EF + Rognan gap", gap_val, epoch)

            efs, *_ = run_virtual_screen(model, test_vs_loader, lower_is_better=lower_is_better)
            writer.add_scalar("Test EF", np.mean(efs), epoch)

            if monitor_robin:
                robin_ef05 = robin_eval(cfg, model)
                writer.add_scalar("Robin EF", robin_ef05, epoch)

        # Finally do checkpointing based on vs performance, we negate it since higher efs are better
        # loss_to_track = val_loss
        if val_vs_loader_rognan is not None and "monitor_gap" in cfg.train and cfg.train.monitor_gap:
            loss_to_track = -gap_val
        else:
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
