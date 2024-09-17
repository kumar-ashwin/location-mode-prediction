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
    x, y, x_dict, y_mode = inputs
    if config.networkName == "deepmove":
        x = (x[0].to(device), x[1].to(device))
        
        for key in x_dict[0]:
            x_dict[0][key] = x_dict[0][key].to(device)
        for key in x_dict[1]:
            x_dict[1][key] = x_dict[1][key].to(device)
    else:
        x = x.to(device)
        for key in x_dict:
            x_dict[key] = x_dict[key].to(device)
    y = y.to(device)
    y_mode = y_mode.to(device)

    return x, y, x_dict, y_mode


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

        x, y, x_dict, y_mode = send_to_device(inputs, device, config)

        for key, value in x_dict.items():
            if torch.isnan(value).any():
                print(f"NaN detected in input {key}")
        if torch.isnan(x).any():
            print("NaN detected in x")


        if config.if_loss_mode:
            if config.if_embed_next_mode:
                logits_loc, logits_mode = model(x, x_dict, device, next_mode=y_mode)
            else:
                logits_loc, logits_mode = model(x, x_dict, device)

            if torch.isnan(logits_loc).any():
                print("Nan detected in logits_loc")
                print("X: ", x)
                print("logits_loc: ", logits_loc[:10])
                print("logits_mode: ", logits_mode[:10])
                print("y: ", y)
                print("y_mode: ", y_mode)
                sys.exit()
            loss_size_loc = CEL(logits_loc, y.reshape(-1))
            loss_size_mode = CEL(logits_mode, y_mode.reshape(-1))
            
            # check if the loss is nan
            if torch.isnan(loss_size_loc) or torch.isnan(loss_size_mode):
                print("Nan detected in loss")
                print("loss_size_loc: ", loss_size_loc)
                print("loss_size_mode: ", loss_size_mode)
                print("logits_loc: ", logits_loc[:10])
                print("logits_mode: ", logits_mode[:10])
                print("y: ", y.reshape(-1))
                print("y_mode: ", y_mode.reshape(-1))
                sys.exit()
            loss_size = loss_size_loc + loss_size_mode
        else:
            logits_loc = model(x, x_dict, device)
            loss_size = CEL(logits_loc, y.reshape(-1))

        # print("Loss size: ", loss_size)

        optim.zero_grad()
        # Check if loss tries to access illegal memory. Safeguard
        if torch.isnan(loss_size):
            print("Nan detected in loss item. Skip backward")
            optim.zero_grad()
            continue
        else:
            try:
                loss_size.backward()
            except:
                print("Some issue arised in backward pass. Skip backward")
                print("logits_loc: ", logits_loc[:10])
                print("logits_mode: ", logits_mode[:10])
                print("y: ", y)
                print("y_mode: ", y_mode)
                # clear the loss

        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print("NaN detected in gradients.")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
        if scheduler_count == 0:
            scheduler.step()

        # Print statistics
        # #Chcek if the loss item is nan
        if torch.isnan(loss_size):
            print("Nan detected in loss item. Skip adding to running loss")
            print("logits_loc: ", logits_loc[:10])
            # print("logits_mode: ", logits_mode[:10])
            print("y: ", y)
            print("y_mode: ", y_mode)
        else:
            running_loss += loss_size.item()


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
            if (i + 1) % (n_batches // 100) == 0:
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

    # def debug_forward(self, out, user, mode_emb=None, next_mode=None) -> Tensor:
    #     # To trace where nans appear in the model

    #     # with fc output
    #     if self.if_embed_user:
    #         emb = self.emb_user(user)

    #         if self.if_embed_next_mode:
    #             emb += self.next_mode_fc(mode_emb(next_mode))

    #         out = torch.cat([out, emb], -1)
        
    #     print(out)
        
    #     out = self.emb_dropout(out)

    #     # residual
    #     if self.if_residual_layer:
    #         out = self.norm_1(out + self.fc_dropout(F.relu(self.fc_1(out))))

    #     print('After residual:', out)
        
    #     if self.if_loss_mode:
    #         return self.fc_loc(out), self.fc_mode(out), out
    #     else:
    #         return self.fc_loc(out), out
def validate(config, model, data_loader, device):

    total_val_loss = 0
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    CEL = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for inputs in data_loader:

            x, y, x_dict, y_mode = send_to_device(inputs, device, config)
            # print("X: ", x)
            if config.if_loss_mode:
                if config.if_embed_next_mode:
                    logits_loc, logits_mode = model(x, x_dict, device, next_mode=y_mode)
                else:
                    logits_loc, logits_mode = model(x, x_dict, device)

                loss_size_loc = CEL(logits_loc, y.reshape(-1))
                loss_size_mode = CEL(logits_mode, y_mode.reshape(-1))

                # check if the loss is nan
                if torch.isnan(loss_size_loc) or torch.isnan(loss_size_mode):
                    print("Nan detected in validation loss")
                    print("loss_size_loc: ", loss_size_loc)
                    print("loss_size_mode: ", loss_size_mode)
                    print("logits_loc: ", logits_loc[:10])
                    print("logits_mode: ", logits_mode[:10])
                    print("y: ", y)
                    print("y_mode: ", y_mode)

                    print("Debug forward")
                    model.debug_forward(x, x_dict, device, next_mode=y_mode)
                    exit()
                    

                loss_size = loss_size_loc + loss_size_mode
            else:
                logits_loc = model(x, x_dict, device)
                loss_size = CEL(logits_loc, y.reshape(-1))
                #check if the loss is nan
                if torch.isnan(loss_size):
                    print("Nan detected in validation loss (size)")
                    print("X: ", x)
                    print("logits_loc: ", logits_loc[:10])
                    print("y: ", y)
                    print("loss size: ", loss_size)
            

            if torch.isnan(loss_size):
                print("Nan detected in validation loss item. Skip adding to running loss")
                exit()
            else:
                total_val_loss += loss_size.item()

            result_arr += calculate_correct_total_prediction(logits_loc, y.view(-1))

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
        for j in range(1, 8):
            result_dict[i][j] = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            batch_dict[i][j] = 0

    # change to validation mode
    model.eval()
    with torch.no_grad():

        for inputs in data_loader:
            x, y, x_dict, y_mode = send_to_device(inputs, device, config)

            if config.if_loss_mode:
                if config.if_embed_next_mode:
                    logits_loc, _ = model(x, x_dict, device, next_mode=y_mode)
                else:
                    logits_loc, _ = model(x, x_dict, device)
            else:
                logits_loc = model(x, x_dict, device)

            # we get the per user per mode accuracy
            user_arr = x_dict["user"].cpu().detach().numpy()
            mode_arr = y_mode.cpu().detach().numpy()
            for user in np.unique(user_arr):
                # index belong to the current user 
                user_index = np.nonzero(user_arr == user)[0]
                for mode in np.unique(mode_arr):
                    mode_index = np.nonzero(mode_arr == mode)[0]
                    index = set(user_index).intersection(set(mode_index))
                    if not len(index):
                        continue
                    index = np.array(list(index))
                    result_dict[user][mode] += calculate_correct_total_prediction(logits_loc[index, :], y[index])
                    batch_dict[user][mode] += 1

            result_arr += calculate_correct_total_prediction(logits_loc, y.view(-1))

    # f1 score
    for i in range(1, config.total_user_num):
        for j in range(1, 8):
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
