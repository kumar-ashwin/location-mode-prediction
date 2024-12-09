import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import DatasetDict, Dataset, load_from_disk

import os
import torch
from torch.nn.utils.rnn import pad_sequence

class geotron_dataset(torch.utils.data.Dataset):
    """
    Dataset class for next poi prediction.
    Uses data saved as arrow files (huggingface Datasets).
    """
    def __init__(
        self,
        source_root,
        dataset='',
        data_type='train',
        model_type='transformer',
        max_seq_len=100,
    ):
        self.root = source_root
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        self.model_type = model_type
        self.data_dir = os.path.join(source_root, 'processed/')
        self.save_path = os.path.join(self.data_dir, f"{self.dataset}/processed_dataset")
        print(f'Initializing dataset from {self.save_path}.')

        # load the dataset with 'data_type' split
        self.data = load_from_disk(self.save_path)
        self.data = self.data[data_type]
        self.len = len(self.data)
        print(f'Loaded {data_type} dataset with {self.len} samples.')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        data_dict["X"] = hist["poi_enc"].values
        data_dict["start_min_X"] = hist["start_min"].values
        data_dict["weekday_X"] = hist["weekday"].values
        data_dict["cluster_X"] = hist["cluster"].values
        data_dict["intra_cluster_id_X"] = hist["intra_cluster_id"].values
        # data_dict["time_to_next_X"] = hist["time_to_next"].values
        
        # add context features: homelat, homelon, time to next
        data_dict["user"] = row["user_id_enc"]
        data_dict["homelat"] = row["homelat"]
        data_dict["homelon"] = row["homelon"]
        data_dict["time_to_next"] = hist["time_to_next"].values[-1]

        # the next poi is the Y (combine with cluster and intra_cluster_id)
        data_dict["poi_Y"] = int(row["poi"])
        data_dict["cluster_Y"] = int(row["cluster"])
        data_dict["intra_cluster_id_Y"] = int(row["intra_cluster_id"])
        """
        sample = self.data[idx]
        x = torch.tensor(sample['X'], dtype=torch.int64)

        x_dict = {}
        # Time series features
        # x_dict["time"] = torch.tensor(sample["start_min_X"], dtype=torch.int32) // 15
        x_dict["time"] = torch.div(torch.tensor(sample["start_min_X"], dtype=torch.int32), 15, rounding_mode="floor")
        x_dict["weekday"] = torch.tensor(sample["weekday_X"], dtype=torch.int32)
        x_dict["cluster"] = torch.tensor(sample["cluster_X"], dtype=torch.int32)
        x_dict["intra_cluster_id"] = torch.tensor(sample["intra_cluster_id_X"], dtype=torch.int32)
        # Context features. Not vectors.
        x_dict["user"] = torch.tensor(sample["user"], dtype=torch.int32)
        x_dict["homelat"] = torch.tensor(sample["homelat"], dtype=torch.float32)
        x_dict["homelon"] = torch.tensor(sample["homelon"], dtype=torch.float32)
        # x_dict["time_to_next"] = torch.tensor(sample["time_to_next"], dtype=torch.int32)//84600  # Num seconds in day 
        x_dict["time_to_next"] = torch.div(torch.tensor(sample["time_to_next"], dtype=torch.int32), 84600, rounding_mode="floor")  # Num seconds in day

        if len(x) > self.max_seq_len:
            x = x[-self.max_seq_len:]
            x_dict["time"] = x_dict["time"][-self.max_seq_len:]
            x_dict["weekday"] = x_dict["weekday"][-self.max_seq_len:]
            x_dict["cluster"] = x_dict["cluster"][-self.max_seq_len:]
            x_dict["intra_cluster_id"] = x_dict["intra_cluster_id"][-self.max_seq_len:]
            
        y = torch.tensor(sample["poi_Y"], dtype=torch.int64)
        y_cluster = torch.tensor(sample["cluster_Y"], dtype=torch.int64)
        y_intra_cluster_id = torch.tensor(sample["intra_cluster_id_Y"], dtype=torch.int64)

        return x, y, x_dict, y_cluster, y_intra_cluster_id


def collate_fn(batch):
    """function to collate data samples into batch tensors."""
    x_batch, y_batch, y_cluster_batch, y_intra_cluster_id_batch = [], [], [], []
    x_dict_batch = {"len": []}
    for key in batch[0][2]:
        x_dict_batch[key] = []

    for x_sample, y_sample, x_dict_sample, y_cluster_sample, y_intra_cluster_id_sample in batch:
        x_batch.append(x_sample)
        y_batch.append(y_sample)
        y_cluster_batch.append(y_cluster_sample)
        y_intra_cluster_id_batch.append(y_intra_cluster_id_sample)

        # x_dict_sample
        x_dict_batch["len"].append(len(x_sample))
        for key in x_dict_sample:
            x_dict_batch[key].append(x_dict_sample[key])

    x_batch = pad_sequence(x_batch)
    y_batch = torch.tensor(y_batch, dtype=torch.int64)
    y_cluster_batch = torch.tensor(y_cluster_batch, dtype=torch.int64)
    y_intra_cluster_id_batch = torch.tensor(y_intra_cluster_id_batch, dtype=torch.int64)

    # x_dict_batch
    non_vector_keys = ["user", "homelat", "homelon", "time_to_next"]
    for key in x_dict_batch:
        if key in non_vector_keys:
            x_dict_batch[key] = torch.tensor(x_dict_batch[key], dtype=torch.float32)
        elif key == "len":
            x_dict_batch["len"] = torch.tensor(x_dict_batch["len"], dtype=torch.int64)
        else:
            x_dict_batch[key] = pad_sequence(x_dict_batch[key])

    return x_batch, y_batch, x_dict_batch, y_cluster_batch, y_intra_cluster_id_batch


class geotron_eval_dataset(torch.utils.data.Dataset):
    """
    Dataset class for evaluation sequences, for topk prediction accuracy in the next 1w, 2w, 1m
    """
    def __init__(
        self,
        source_root,
        dataset='',
        max_seq_len=200,
    ):
        self.root = source_root
        self.dataset = dataset
        self.max_seq_len = max_seq_len
        
        self.data_dir = os.path.join(source_root, 'processed/')
        self.save_path = os.path.join(self.data_dir, f"{self.dataset}/processed_sequences_with_future_weeks")
        print(f'Initializing dataset from {self.save_path}.')

        # load the dataset with 'data_type' split
        self.data = load_from_disk(self.save_path)
        self.data = self.data['test']
        self.len = len(self.data)
        print(f'Loaded eval dataset with {self.len} samples.')

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # Similar to geotron_dataset
        # Instead of single ys, have poi_Y1w, cluster_Y1w, intra_cluster_id_Y1w, similar for Y2w, Y1m
        sample = self.data[idx]
        x = torch.tensor(sample['X'], dtype=torch.int64)

        x_dict = {}
        # Time series features
        x_dict["time"] = torch.div(torch.tensor(sample["start_min_X"], dtype=torch.int32), 15, rounding_mode="floor")
        x_dict["weekday"] = torch.tensor(sample["weekday_X"], dtype=torch.int32)
        x_dict["cluster"] = torch.tensor(sample["cluster_X"], dtype=torch.int32)
        x_dict["intra_cluster_id"] = torch.tensor(sample["intra_cluster_id_X"], dtype=torch.int32)
        
        # Context features. Not vectors.
        x_dict["user"] = torch.tensor(sample["user"], dtype=torch.int32)
        x_dict["homelat"] = torch.tensor(sample["homelat"], dtype=torch.float32)
        x_dict["homelon"] = torch.tensor(sample["homelon"], dtype=torch.float32)
        x_dict["time_to_next"] = torch.div(torch.tensor(sample["time_to_next"], dtype=torch.int32), 84600, rounding_mode="floor")  # Num seconds in day

        # check if the sequence is longer than max_seq_len
        if len(x) > self.max_seq_len:
            x = x[-self.max_seq_len:]
            x_dict["time"] = x_dict["time"][-self.max_seq_len:]
            x_dict["weekday"] = x_dict["weekday"][-self.max_seq_len:]
            x_dict["cluster"] = x_dict["cluster"][-self.max_seq_len:]
            x_dict["intra_cluster_id"] = x_dict["intra_cluster_id"][-self.max_seq_len:]

        y1w_dict = {}
        y1w_dict["poi_Y"] = torch.tensor(sample["poi_Y1w"], dtype=torch.int64)
        y1w_dict["cluster_Y"] = torch.tensor(sample["cluster_Y1w"], dtype=torch.int64)
        y1w_dict["intra_cluster_id_Y"] = torch.tensor(sample["intra_cluster_id_Y1w"], dtype=torch.int64)

        y2w_dict = {}
        y2w_dict["poi_Y"] = torch.tensor(sample["poi_Y2w"], dtype=torch.int64)
        y2w_dict["cluster_Y"] = torch.tensor(sample["cluster_Y2w"], dtype=torch.int64)
        y2w_dict["intra_cluster_id_Y"] = torch.tensor(sample["intra_cluster_id_Y2w"], dtype=torch.int64)

        y1m_dict = {}
        y1m_dict["poi_Y"] = torch.tensor(sample["poi_Y1m"], dtype=torch.int64)
        y1m_dict["cluster_Y"] = torch.tensor(sample["cluster_Y1m"], dtype=torch.int64)
        y1m_dict["intra_cluster_id_Y"] = torch.tensor(sample["intra_cluster_id_Y1m"], dtype=torch.int64)

        return x, x_dict, y1w_dict, y2w_dict, y1m_dict


def collate_fn_eval(batch):
    """function to collate data samples into batch tensors."""
    x_batch = []
    x_dict_batch = {"len": []}
    for key in batch[0][1]:
        x_dict_batch[key] = []

    y1w_batch, y2w_batch, y1m_batch = {}, {}, {}
    ys = [y1w_batch, y2w_batch, y1m_batch]
    for y_dict in ys:
        for key in batch[0][2]:
            y_dict[key] = []

    for x_sample, x_dict_sample, y1w_dict, y2w_dict, y1m_dict in batch:
        x_batch.append(x_sample)

        x_dict_batch["len"].append(len(x_sample))
        for key in x_dict_sample:
            x_dict_batch[key].append(x_dict_sample[key])

        for y_dict, y_dict_sample in zip(ys, [y1w_dict, y2w_dict, y1m_dict]):
            for key in y_dict_sample:
                y_dict[key].append(y_dict_sample[key])

    x_batch = pad_sequence(x_batch)

    # x_dict_batch
    non_vector_keys = ["user", "homelat", "homelon", "time_to_next"]
    for key in x_dict_batch:
        if key in non_vector_keys:
            x_dict_batch[key] = torch.tensor(x_dict_batch[key], dtype=torch.float32)
        elif key == "len":
            x_dict_batch["len"] = torch.tensor(x_dict_batch["len"], dtype=torch.int64)
        else:
            x_dict_batch[key] = pad_sequence(x_dict_batch[key])

    return x_batch, x_dict_batch, y1w_batch, y2w_batch, y1m_batch

def test_dataloader(train_loader):
    ave_shape = 0
    hist_shape = 0
    y_shape = 0
    y_cluster_shape = 0
    y_intra_cluster_id_shape = 0
    user_ls = []
    for batch_idx, (x, y, x_dict, y_cluster, y_intra_cluster_id) in tqdm(enumerate(train_loader)):
        # print("batch_idx ", batch_idx)
        # print(inputs.shape)
        print(x)
        (hist, x) = x
        (hist_dict, x_dict) = x_dict

        hist_shape += hist.shape[0]
        ave_shape += x.shape[0]
        y_shape += y.shape[0]
        y_cluster_shape += y_cluster.shape[0]
        y_intra_cluster_id_shape += y_intra_cluster_id.shape[0]

        user_ls.extend(x_dict["user"])

    # print(np.max(user_ls), np.min(user_ls))
    print(hist_shape / len(train_loader))
    print(ave_shape / len(train_loader))
    print(y_shape / len(train_loader))
    print(y_cluster_shape / len(train_loader))
    print(y_intra_cluster_id_shape / len(train_loader))

if __name__ == "__main__":
    source_root = r"./data/"

    dataset_train = geotron_dataset(
        source_root, data_type="train", dataset="Eastland", model_type="transformer"
    )
    dataset_val = geotron_dataset(
        source_root, data_type="val", dataset="Eastland", model_type="transformer"
    )
    dataset_test = geotron_dataset(
        source_root, data_type="test", dataset="Eastland", model_type="transformer"
    )

    kwds = {"shuffle": True, "num_workers": 0, "batch_size": 2}
    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, **kwds)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=collate_fn, **kwds)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=collate_fn, **kwds)

    test_dataloader(train_loader)
    test_dataloader(val_loader)
    test_dataloader(test_loader)
