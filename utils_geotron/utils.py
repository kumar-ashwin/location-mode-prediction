# from msilib.schema import Error
import yaml
import random, torch, os
import numpy as np
import pandas as pd

from utils_geotron.train import trainNet, test, get_performance_dict

from utils_geotron.dataloader_new import geotron_dataset, collate_fn

from models.model_geotron import TransEncoder, SimpleEncoder, LSTMEncoder
from models.RNNs import RNNs


def load_config(path):
    """Load config file.
    Args:
        path (str): path to the config file
    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config


def setup_seed(seed):
    """Fix random seed for deterministic training."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_trainedNets(config, model, train_loader, val_loader, device, log_dir):

    best_model, performance = trainNet(config, model, train_loader, val_loader, device, log_dir=log_dir)
    performance["type"] = "vali"

    return best_model, performance


def get_test_result(config, best_model, test_loader, device):
    return_dict, user_cluster_dict = test(config, best_model, test_loader, device)
    performance = get_performance_dict(return_dict)
    performance["type"] = "test"

    user_mode_ls = []
    for user_id, mode_dict in user_cluster_dict.items():
        for mode_id, perf in mode_dict.items():
            # ids and the count of records
            user_mode_ls.append(np.append(perf, np.array([user_id, mode_id])))

    result_df = pd.DataFrame(user_mode_ls)
    result_df.columns = ["correct@1", "correct@3", "correct@5", "correct@10", "f1", "rr", "total", "user_id", "mode_id"]

    return performance, result_df


def get_models(config, device):
    total_params = 0

    if config.networkName == "transformer":
        model = TransEncoder(config=config).to(device)
    elif config.networkName == "mlp":
        model = SimpleEncoder(config=config).to(device)
    elif config.networkName == "lstm":
        model = LSTMEncoder(config=config).to(device)
    else:
        raise ValueError("Invalid network name")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total number of trainable parameters: ", total_params)

    return model


def get_dataloaders(config):

    kwds_train = {
        "shuffle": True,
        "num_workers": 0,
        "batch_size": config["batch_size"],
    }
    kwds_val = {
        "shuffle": False,
        "num_workers": 0,
        "batch_size": config["batch_size"],
    }
    kwds_test = {
        "shuffle": False,
        "num_workers": 0,
        "batch_size": config["batch_size"],
    }

    dataset_train = geotron_dataset(
        config.source_root,
        data_type="train",
        model_type=config.networkName,
        dataset=config.dataset,
    )
    dataset_val = geotron_dataset(
        config.source_root,
        data_type="val",
        model_type=config.networkName,
        dataset=config.dataset,
    )
    dataset_test = geotron_dataset(
        config.source_root,
        data_type="test",
        model_type=config.networkName,
        dataset=config.dataset,
    )

    fn = collate_fn

    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=fn, **kwds_train)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=fn, **kwds_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=fn, **kwds_test)

    # print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader
