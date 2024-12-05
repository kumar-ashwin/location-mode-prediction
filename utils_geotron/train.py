import sys, os
import pandas as pd
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score

import time

from transformers import get_linear_schedule_with_warmup

from utils.earlystopping import EarlyStopping
from utils.dataloader import load_pk_file


def get_performance_dict(return_dict):
    perf = {
        "correct@1": return_dict["correct@1"],
        # "correct@3": return_dict["correct@3"],
        "correct@20": return_dict["correct@20"],
        "correct@5": return_dict["correct@5"],
        "correct@10": return_dict["correct@10"],
        # "correct@20": return_dict["correct@20"],
        "rr": return_dict["rr"],
        "f1": return_dict["f1"],
        "total": return_dict["total"],
    }

    perf["acc@1"] = perf["correct@1"] / perf["total"] * 100
    perf["acc@5"] = perf["correct@5"] / perf["total"] * 100
    perf["acc@10"] = perf["correct@10"] / perf["total"] * 100
    perf["acc@20"] = perf["correct@20"] / perf["total"] * 100
    perf["mrr"] = perf["rr"] / perf["total"] * 100

    return perf


def send_to_device(inputs, device, config):
    x, y, x_dict, y_cluster, y_intra_cluster_id = inputs
    x = x.to(device)
    for key in x_dict:
        x_dict[key] = x_dict[key].to(device)
    y = y.to(device)
    y_cluster = y_cluster.to(device)
    y_intra_cluster_id = y_intra_cluster_id.to(device)

    return x, y, x_dict, y_cluster, y_intra_cluster_id


def calculate_correct_total_prediction(logits, true_y):

    # top_ = torch.eq(torch.argmax(logits, dim=-1), true_y).sum().cpu().numpy()

    result_ls = []
    # for k in [1, 3, 5, 10]:
    for k in [1, 20, 5, 10]:
        if logits.shape[-1] < k:
            prediction = torch.argmax(logits, dim=-1)
        else:
            prediction = torch.topk(logits, k=k, dim=-1).indices
        # f1 score
        if k == 1:
            f1 = f1_score(true_y.cpu(), prediction.cpu(), average="weighted")

        top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum().cpu().numpy()
        # top_k = np.sum([curr_y in pred for pred, curr_y in zip(prediction, true_y)])
        result_ls.append(top_k)

    # f1 score
    result_ls.append(f1)
    # mrr
    result_ls.append(get_mrr(logits, true_y))
    # total
    result_ls.append(true_y.shape[0])

    return np.array(result_ls, dtype=np.float32)

def calculate_correct_total_prediction_tiered(logits_cluster, true_y_cluster, logits_icid, true_y_icid):
    # A result is correct if both the cluster and the intra cluster id are correct

    # Top-k prediction: pick the combination of 

    result_ls = []
    batch_size = logits_cluster.shape[0]

    for current_k in [1, 20, 5, 10]:
        # # Prune to top-k for CID
        # cid_probs, top_cids = torch.topk(torch.softmax(logits_cluster, dim=-1), k=current_k, dim=-1)  # (batch_size, k)

        # # Prune to top-k for ICID
        # icid_probs, top_icids = torch.topk(torch.softmax(logits_icid, dim=-1), k=current_k, dim=-1)  # (batch_size, k)

        # # Compute joint probabilities within the reduced Cartesian product
        # joint_correct_count = 0
        # for i in range(batch_size):
        #     # Get the Cartesian product of top-k CIDs and top-k ICIDs
        #     cid_prob = cid_probs[i].unsqueeze(1)  # (k, 1)
        #     icid_prob = icid_probs[i].unsqueeze(0)  # (1, k)

        #     joint_prob = (cid_prob * icid_prob).flatten()  # Joint probabilities (k*k,)
        #     joint_indices = torch.cartesian_prod(top_cids[i], top_icids[i])  # All (CID, ICID) pairs (k*k, 2)

        #     # Find the top-k joint probabilities
        #     top_joint_indices = joint_indices[torch.topk(joint_prob, k=current_k).indices]  # (k, 2)

        #     # Check if the true pair is in the top-k joint predictions
        #     true_pair = torch.tensor([true_y_cluster[i].item(), true_y_icid[i].item()])
        #     if any(torch.equal(true_pair, pred) for pred in top_joint_indices):
        #         joint_correct_count += 1

        # result_ls.append(joint_correct_count)

        # Top-k for CID and ICID
        use_naive_tiered = False
        # !! VARIANT: Use only the top CID
        if use_naive_tiered:
            cid_probs, top_cids = torch.topk(torch.softmax(logits_cluster, dim=-1), k=1, dim=-1)  # (batch_size, 1)
        else:
            cid_probs, top_cids = torch.topk(torch.softmax(logits_cluster, dim=-1), k=current_k, dim=-1)  # (batch_size, k)
        icid_probs, top_icids = torch.topk(torch.softmax(logits_icid, dim=-1), k=current_k, dim=-1)  # (batch_size, k)

        # Compute joint probabilities for all pairs (vectorized)
        joint_probs = cid_probs.unsqueeze(2) * icid_probs.unsqueeze(1)  # (batch_size, k, k)
        joint_probs_flattened = joint_probs.view(batch_size, -1)  # Flatten to (batch_size, k*k)

        # Cartesian product indices for all top-k pairs (vectorized)
        if use_naive_tiered:
            top_cids_expanded = top_cids.unsqueeze(2).expand(-1, -1, current_k)  # (batch_size, 1, k)
            top_icids_expanded = top_icids.unsqueeze(1).expand(-1, 1, current_k)  # (batch_size, 1, k)
        else:
            top_cids_expanded = top_cids.unsqueeze(2).expand(-1, -1, current_k)  # (batch_size, k, k)
            top_icids_expanded = top_icids.unsqueeze(1).expand(-1, current_k, -1)  # (batch_size, k, k)

        joint_indices = torch.stack(
            [top_cids_expanded.reshape(batch_size, -1), top_icids_expanded.reshape(batch_size, -1)], dim=-1
        )  # (batch_size, k*k, 2)

        # Find top-k joint probabilities and their indices
        _, top_joint_indices_flat = torch.topk(joint_probs_flattened, k=current_k, dim=-1)  # (batch_size, k)

        # Gather the corresponding (CID, ICID) pairs for top-k joint probabilities
        top_joint_pairs = torch.gather(
            joint_indices, 1, top_joint_indices_flat.unsqueeze(-1).expand(-1, -1, 2)
        )  # (batch_size, k, 2)

        # Check correctness: Compare true (CID, ICID) pairs with top-k joint pairs
        true_pairs = torch.stack([true_y_cluster, true_y_icid], dim=1).unsqueeze(1)  # (batch_size, 1, 2)
        correct_joint = (true_pairs == top_joint_pairs).all(dim=-1).any(dim=-1)  # (batch_size,)

        # Count correct predictions
        joint_correct_count = correct_joint.sum().item()
        result_ls.append(joint_correct_count)

    # F1-score
    # f1 = f1_score(
    #     torch.stack([true_y_cluster, true_y_icid], dim=1).cpu(),
    #     torch.stack([torch.argmax(logits_cluster, dim=-1), torch.argmax(logits_icid, dim=-1)], dim=1).cpu(),
    #     average="weighted"
    # )
    result_ls.append(0)

    # mrr: not implemented. Return -1 for all
    result_ls.append(-1)

    # Total samples
    result_ls.append(batch_size)

    return np.array(result_ls, dtype=np.float32)

def calculate_correct_total_prediction_naive_tiered(logits_cluster, true_y_cluster, logits_icid, true_y_icid):
    """
    Variant of calculate_correct_total_prediction_tiered that uses a naive approach to find the top-k pairs.
    Selects only the top cid, and then selects top k icids for it. k vs k^2 pairs.
    """
    result_ls = []
    batch_size = logits_cluster.shape[0]

    for current_k in [1, 20, 5, 10]:
        # Top-k for CID and ICID
        cid_probs, top_cids = torch.topk(torch.softmax(logits_cluster, dim=-1), k=1, dim=-1)  # (batch_size, 1)
        icid_probs, top_icids = torch.topk(torch.softmax(logits_icid, dim=-1), k=current_k, dim=-1)  # (batch_size, k)

        # Compute joint probabilities for all pairs (vectorized)
        joint_probs = cid_probs.unsqueeze(2) * icid_probs.unsqueeze(1)  # (batch_size, 1, k)
        joint_probs_flattened = joint_probs.view(batch_size, -1)  # Flatten to (batch_size, k)

        # Cartesian product indices for all top-k pairs (vectorized)
        top_cids_expanded = top_cids.unsqueeze(2).expand(-1, -1, current_k)  # (batch_size, 1, k)
        top_icids_expanded = top_icids.unsqueeze(1).expand(-1, 1, current_k)  # (batch_size, 1, k)

        joint_indices = torch.stack(
            [top_cids_expanded.reshape(batch_size, -1), top_icids_expanded.reshape(batch_size, -1)], dim=-1
        )  # (batch_size, k, 2)

        # Find top-k joint probabilities and their indices
        _, top_joint_indices_flat = torch.topk(joint_probs_flattened, k=current_k, dim=-1)  # (batch_size, k)

        # Gather the corresponding (CID, ICID) pairs for top-k joint probabilities
        top_joint_pairs = torch.gather(
            joint_indices, 1, top_joint_indices_flat.unsqueeze(-1).expand(-1, -1, 2)
        )  # (batch_size, k, 2)

        # Check correctness: Compare true (CID, ICID) pairs with top-k joint pairs
        true_pairs = torch.stack([true_y_cluster, true_y_icid], dim=1).unsqueeze(1)  # (batch_size, 1, 2)
        correct_joint = (true_pairs == top_joint_pairs).all(dim=-1).any(dim=-1)  # (batch_size,)
        # Count correct predictions
        joint_correct_count = correct_joint.sum().item()
        result_ls.append(joint_correct_count)

    # F1-score
    result_ls.append(0)

    # mrr: not implemented. Return -1 for all
    result_ls.append(-1)

    # Total samples
    result_ls.append(batch_size)

    return np.array(result_ls, dtype=np.float32)

def get_mrr(prediction, targets):
    """
    Calculates the MRR score for the given predictions and targets.

    Args:
        prediction (Bxk): torch.LongTensor. the softmax output of the model.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        the sum rr score
    """
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float()
    rranks = torch.reciprocal(ranks)

    return torch.sum(rranks).cpu().numpy()


def get_optimizer(config, model):
    # define the optimizer & learning rate
    if config.optimizer == "SGD":
        optim = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            nesterov=True,
        )
    elif config.optimizer == "Adam":
        optim = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

    return optim


def trainNet(config, model, train_loader, val_loader, device, log_dir):

    performance = {}

    optim = get_optimizer(config, model)

    # define learning rate schedule
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=len(train_loader) * config.num_warmup_epochs,
        num_training_steps=len(train_loader) * config.num_training_epochs,
    )
    scheduler_ES = StepLR(optim, step_size=config.lr_step_size, gamma=config.lr_gamma)
    if config.verbose:
        print("Current learning rate: ", scheduler.get_last_lr()[0])

    #init log.txt
    with open("log.txt", "w") as f:
        f.write("")
    # Time for printing
    training_start_time = time.time()
    globaliter = 0
    scheduler_count = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config["patience"], verbose=config.verbose)

    # Loop for n_epochs
    for epoch in range(config.max_epoch):
        # train for one epoch
        globaliter = train(
            config,
            model,
            train_loader,
            optim,
            device,
            epoch,
            scheduler,
            scheduler_count,
            globaliter,
        )

        # At the end of the epoch, do a pass on the validation set
        return_dict = validate(config, model, val_loader, device)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(return_dict, model)

        if early_stopping.early_stop:
            if config.verbose:
                print("=" * 50)
                print("Early stopping")
            if scheduler_count == 2:
                performance = get_performance_dict(early_stopping.best_return_dict)
                print(
                    "Training finished.\t Time: {:.2f}min.\t acc@1: {:.2f}%".format(
                        (time.time() - training_start_time) / 60,
                        performance["acc@1"],
                    )
                )

                break

            scheduler_count += 1
            model.load_state_dict(torch.load(log_dir + "/checkpoint.pt"))
            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler_ES.step()

        if config.verbose:
            # print("Current learning rate: {:.5f}".format(scheduler.get_last_lr()[0]))
            # print("Current learning rate: {:.5f}".format(scheduler_ES.get_last_lr()[0]))
            print("Current learning rate: {:.5f}".format(optim.param_groups[0]["lr"]))
            print("=" * 50)

        if config.debug == True:
            break

    return model, performance


def train(
    config,
    model,
    train_loader,
    optim,
    device,
    epoch,
    scheduler,
    scheduler_count,
    globaliter,
):
    torch.autograd.set_detect_anomaly(True)
    model.train()

    running_loss = 0.0
    # 1, 3, 5, 10, rr, total
    # 1, 20, 5, 10, rr, total
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    n_batches = len(train_loader)

    CEL = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
    # define start time
    start_time = time.time()
    optim.zero_grad()
    for i, inputs in enumerate(train_loader):
        globaliter += 1

        x, y, x_dict, y_cluster, y_intra_cluster_id = send_to_device(inputs, device, config)

        for key, value in x_dict.items():
            if torch.isnan(value).any():
                print(f"NaN detected in input {key}")
        if torch.isnan(x).any():
            print("NaN detected in x")

        if config.predict_clusters:
            if config.predict_intra_cluster:
                logits_cluster, logits_icid = model(x, x_dict, device)
                loss_size_cluster = CEL(logits_cluster, y_cluster)
                loss_size_icid = CEL(logits_icid, y_intra_cluster_id)
                total_loss = loss_size_cluster + loss_size_icid
                optim.zero_grad()
                total_loss.backward()
            else:
                logits_cluster = model(x, x_dict, device)
                loss_size_cluster = CEL(logits_cluster, y_cluster)
                optim.zero_grad()
                loss_size_cluster.backward()

        else:
            logits_loc = model(x, x_dict, device)
            # print("Target values:", y.reshape(-1))
            # print("Min target value:", y.min().item())
            # print("Max target value:", y.max().item())
            assert torch.min(y) >= 0, f"Labels contain negative values. Min value: {torch.min(y).item()}"
            assert torch.max(y) < logits_loc.shape[1], f"Labels exceed the number of classes. Max value: {torch.max(y).item()}"

            loss_loc = CEL(logits_loc, y.reshape(-1))
            optim.zero_grad()

            try:
                loss_loc.backward()
            except:
                print("Loss value:", loss_loc.item())
                if torch.isnan(logits_loc).any() or torch.isinf(logits_loc).any():
                    # raise ValueError("Logits contain NaN or Inf.")
                    print("Logits contain NaN or Inf!!!")
                #print largest values
                print("Largest values:", torch.topk(logits_loc, 10))
                #print smallest values
                print("Smallest values:", torch.topk(logits_loc, 10, largest=False))

                # Try with smaller batch size to narrow down the issue
                print("Batch size:", x.shape[0])
                x_temp = x.clone()
                y_temp = y.clone()
                x_dict_temp = {key: value.clone() for key, value in x_dict.items()}
                while True:
                    subbatchsize = x_temp.shape[0] // 8
                    if subbatchsize < 1:
                        # Print the data
                        print("Data:")
                        print("x:", x_temp)
                        print("y:", y_temp)
                        print("x_dict:", x_dict_temp)
                    # iterate over the batch size
                    for i in range(8):
                        x_tempi = x_temp[i*subbatchsize:(i+1)*subbatchsize]
                        y_tempi = y_temp[i*subbatchsize:(i+1)*subbatchsize]
                        x_dict_tempi = {key: value[i*subbatchsize:(i+1)*subbatchsize] for key, value in x_dict_temp.items()}

                        # try an update 
                        optim.zero_grad()
                        logits_loc = model(x_tempi, x_dict_tempi, device)
                        loss_loc = CEL(logits_loc, y_tempi.reshape(-1))

                        try:
                            loss_loc.backward()
                        except:
                            print("Found a smaller batch size that causes the issue.")
                            print("Batch size:", half_size)
                            x_temp = x_tempi.clone()
                            y_temp = y_tempi.clone()
                            x_dict_temp = {key: value.clone() for key, value in x_dict_tempi.items()}
                            break
                    print("Unable to find a smaller batch size that causes the issue.")
                    break

                input("Press Enter to continue...")

        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
        if scheduler_count == 0:
            scheduler.step()

        # running_loss += loss_size.item()
        if config.predict_clusters:
            if config.predict_intra_cluster:
                running_loss += loss_size_cluster.item() + loss_size_icid.item()            
                result_arr += calculate_correct_total_prediction_tiered(logits_cluster, y_cluster, logits_icid, y_intra_cluster_id)
            else:
                running_loss += loss_size_cluster.item()
                result_arr += calculate_correct_total_prediction(logits_cluster, y_cluster)
        else:
            running_loss += loss_loc.item()
            result_arr += calculate_correct_total_prediction(logits_loc, y)


        if (config.verbose) and ((i + 1) % config["print_step"] == 0):
            print(
                "Epoch {}, {:.1f}%\t loss: {:.3f} acc@1: {:.2f} f1: {:.2f} mrr: {:.2f}, took: {:.2f}s, acc@5: {:.2f}, acc@10: {:.2f}, acc@20: {:.2f} \r".format(
                    epoch + 1,
                    100 * (i + 1) / n_batches,
                    running_loss / config["print_step"],
                    100 * result_arr[0] / result_arr[-1],
                    100 * result_arr[4] / config["print_step"],
                    100 * result_arr[5] / result_arr[-1],
                    time.time() - start_time,
                    100*result_arr[2] / result_arr[-1],
                    100*result_arr[3] / result_arr[-1],
                    100*result_arr[1] / result_arr[-1],
                ),
                end="",
                flush=True,
            )
            # print(f"Acc@5={result_arr[2]/result_arr[-1]}, \t Acc@10={result_arr[3]/result_arr[-1]}")
            # every 1%, write the output to log.txt (append)
            num_steps = min(100, n_batches)
            if (i + 1) % (n_batches // num_steps) == 0:
                with open("log.txt", "a") as f:
                    f.write(
                        "Epoch {}, {:.1f}%\t loss: {:.3f} acc@1: {:.2f} f1: {:.2f} mrr: {:.2f}, took: {:.2f}s, acc@5: {:.2f}, acc@10: {:.2f}, acc@20: {:.2f} \n".format(
                            epoch + 1,
                            100 * (i + 1) / n_batches,
                            running_loss / config["print_step"],
                            100 * result_arr[0] / result_arr[-1],
                            100 * result_arr[4] / config["print_step"],
                            100 * result_arr[5] / result_arr[-1],
                            time.time() - start_time,
                            100*result_arr[2] / result_arr[-1],
                            100*result_arr[3] / result_arr[-1],
                            100*result_arr[1] / result_arr[-1],
                        )
                    )


            # Reset running loss and time
            running_loss = 0.0
            result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            start_time = time.time()

        if (config["debug"] == True) and (i > 20):
            break
    if config.verbose:
        print()
    return globaliter


def validate(config, model, data_loader, device):

    total_val_loss = 0
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    CEL = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for inputs in data_loader:

            x, y, x_dict, y_cluster, y_intra_cluster_id = send_to_device(inputs, device, config)
            
            if config.predict_clusters:
                if config.predict_intra_cluster:
                    logits_cluster, logits_icid = model(x, x_dict, device)
                    loss_size_cluster = CEL(logits_cluster, y_cluster)
                    loss_size_icid = CEL(logits_icid, y_intra_cluster_id)
                    total_val_loss += loss_size_cluster.item() + loss_size_icid.item()
                    result_arr += calculate_correct_total_prediction_tiered(logits_cluster, y_cluster, logits_icid, y_intra_cluster_id)
                else:
                    logits_cluster = model(x, x_dict, device)
                    loss_size_cluster = CEL(logits_cluster, y_cluster)
                    total_val_loss += loss_size_cluster.item()
                    result_arr += calculate_correct_total_prediction(logits_cluster, y_cluster)
            else:
                logits_loc = model(x, x_dict, device)
                loss_loc = CEL(logits_loc, y.reshape(-1))
                total_val_loss += loss_loc.item()
                result_arr += calculate_correct_total_prediction(logits_loc, y)

    val_loss = total_val_loss / len(data_loader)
    result_arr[4] = result_arr[4] / len(data_loader)

    if config.verbose:
        print(
            "Validation loss = {:.2f} acc@1 = {:.2f} f1 = {:.2f} mrr = {:.2f}".format(
                val_loss,
                100 * result_arr[0] / result_arr[-1],
                100 * result_arr[4],
                100 * result_arr[5] / result_arr[-1],
            ),
        )
        print(
            "acc@5 = {:.2f} acc@10 = {:.2f} acc@20 = {:.2f}".format(
                100 * result_arr[2] / result_arr[-1],
                100 * result_arr[3] / result_arr[-1],
                100 * result_arr[1] / result_arr[-1],
            ),
        )

    return {
        "val_loss": val_loss,
        "correct@1": result_arr[0],
        # "correct@3": result_arr[1],
        "correct@20": result_arr[1],
        "correct@5": result_arr[2],
        "correct@10": result_arr[3],
        "f1": result_arr[4],
        "rr": result_arr[5],
        "total": result_arr[6],
    }


def test(config, model, data_loader, device):
    # overall accuracy
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    # per user accuracy
    result_dict = {}
    batch_dict = {}
    for i in range(1, config.total_user_num):
        result_dict[i] = {}
        batch_dict[i] = {}
        for j in range(1,8): # TODO: use config.total_cluster_num
            result_dict[i][j] = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            batch_dict[i][j] = 0

    # change to validation mode
    model.eval()
    with torch.no_grad():

        for inputs in data_loader:
            x, y, x_dict, y_cluster, y_intra_cluster_id = send_to_device(inputs, device, config)

            if config.predict_clusters:
                if config.predict_intra_cluster:
                    logits_cluster, logits_icid = model(x, x_dict, device)
                    result_arr += calculate_correct_total_prediction_tiered(logits_cluster, y_cluster, logits_icid, y_intra_cluster_id)
                else:
                    logits_cluster = model(x, x_dict, device)
                    result_arr += calculate_correct_total_prediction(logits_cluster, y_cluster)
            else:
                logits_loc = model(x, x_dict, device)
                result_arr += calculate_correct_total_prediction(logits_loc, y)

            # # we get the per user per cluster accuracy
            # user_arr = x_dict["user"].cpu().detach().numpy()
            # cluster_arr = y_cluster.cpu().detach().numpy()
            # # select the correct (relevant) logits and y
            # if config.predict_clusters:
            #     logits_rel = logits_cluster
            #     y_rel = y_cluster
            # else:
            #     logits_rel = logits_loc
            #     y_rel = y
            # for user in np.unique(user_arr):
            #     # index belong to the current user 
            #     user_index = np.nonzero(user_arr == user)[0]
                
            #     for cluster in np.unique(cluster_arr):
            #         cluster_index = np.nonzero(cluster_arr == mode)[0]
            #         index = set(user_index).intersection(set(cluster_index))
            #         if not len(index):
            #             continue
            #         index = np.array(list(index))
            #         result_dict[user][cluster] += calculate_correct_total_prediction(logits_rel[index, :], y_rel[index])

            # result_arr += calculate_correct_total_prediction(logits_loc, y.view(-1))

    # f1 score
    for i in range(1, config.total_user_num):
        for j in range(1, 8): # TODO: use config.total_cluster_num
            if batch_dict[i][j] != 0:
                result_dict[i][j][4] = result_dict[i][j][4]/batch_dict[i][j]

    result_arr[4] = result_arr[4] / len(data_loader)

    if config.verbose:
        print(
            "acc@1 = {:.2f} f1 = {:.2f} mrr = {:.2f}".format(
                100 * result_arr[0] / result_arr[-1],
                100 * result_arr[4],
                100 * result_arr[5] / result_arr[-1],
            ),
        )
        print(
            "acc@5 = {:.2f} acc@10 = {:.2f} acc@20 = {:.2f}".format(
                100 * result_arr[2] / result_arr[-1],
                100 * result_arr[3] / result_arr[-1],
                100 * result_arr[1] / result_arr[-1],
            ),
        )

    return (
        {
            "correct@1": result_arr[0],
            # "correct@3": result_arr[1],
            "correct@20": result_arr[1],
            "correct@5": result_arr[2],
            "correct@10": result_arr[3],
            "f1": result_arr[4],
            "rr": result_arr[5],
            "total": result_arr[6],
        },
        result_dict
    )
