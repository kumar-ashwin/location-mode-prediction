import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
import pickle as pickle
from shapely import wkt

from joblib import Parallel, delayed
from sklearn.preprocessing import OrdinalEncoder
import os
import torch
from torch.nn.utils.rnn import pad_sequence

import trackintel as ti

import h5py

# class sp_loc_dataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         source_root,
#         dataset="gc",
#         data_type="train",
#         previous_day=7,
#         model_type="transformer",
#     ):
#         self.root = source_root
#         self.data_type = data_type
#         self.previous_day = previous_day
#         self.model_type = model_type
#         self.dataset = dataset

#         self.data_dir = os.path.join(source_root, "temp")
#         save_path = os.path.join(
#             self.data_dir,
#             f"{self.dataset}_{self.model_type}_{self.previous_day}_preprocessed.h5",
#         )

#         print(f"Loading data from {save_path}.")
#         if Path(save_path).is_file():
#             with h5py.File(save_path, "r") as hdf:
#                 self.data = {key: hdf[f"{self.data_type}/{key}"][:] for key in hdf[f"{self.data_type}"].keys()}
#         else:
#             print("Please generate preprocessed data using script (preprocess_h5_data.py).")
#             exit()

#         self.len = self.data["X"].shape[0]  # Update to reflect the length of the loaded subset

#     def __len__(self):
#         """Return the length of the current dataloader."""
#         return self.len

#     def __getitem__(self, idx):
#         selected = {key: self.data[key][idx] for key in self.data.keys()}

#         # Assuming the padded sequences have a consistent shape and you want to handle them correctly
#         x = torch.tensor(selected["X"], dtype=torch.int64)

#         x_dict = {}
#         x_dict["mode"] = torch.tensor(selected["mode_X"], dtype=torch.int64)
#         x_dict["user"] = torch.tensor(selected["user_X"][0], dtype=torch.int64)
#         x_dict["time"] = torch.tensor(selected["start_min_X"] // 15, dtype=torch.int64)
#         x_dict["length"] = torch.log(torch.tensor(selected["length_X"], dtype=torch.float32))
#         x_dict["weekday"] = torch.tensor(selected["weekday_X"], dtype=torch.int64)

#         if self.model_type == "deepmove":
#             x_dict["history_count"] = torch.tensor(selected["history_count"])

#         # Handle padding: if sequences are padded, remove the padding for specific processing
#         # For example, if padding value is 0, and you want to remove trailing padding:
#         if x.size(0) > 0:
#             non_padded_indices = (x != 0).nonzero(as_tuple=True)[0]
#             x = x[non_padded_indices]
#             x_dict["mode"] = x_dict["mode"][non_padded_indices]
#             x_dict["time"] = x_dict["time"][non_padded_indices]
#             x_dict["length"] = x_dict["length"][non_padded_indices]
#             x_dict["weekday"] = x_dict["weekday"][non_padded_indices]

#         y = torch.tensor(selected["loc_Y"], dtype=torch.long)
#         y_mode = torch.tensor(selected["mode_Y"], dtype=torch.long)

#         return x, y, x_dict, y_mode
# class sp_loc_dataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         source_root,
#         dataset="gc",
#         data_type="train",
#         previous_day=7,
#         model_type="transformer",
#     ):
#         self.root = source_root
#         self.data_type = data_type
#         self.previous_day = previous_day
#         self.model_type = model_type
#         self.dataset = dataset

#         self.data_dir = os.path.join(source_root, "temp")
#         self.save_path = os.path.join(
#             self.data_dir,
#             f"{self.dataset}_{self.model_type}_{self.previous_day}_preprocessed.h5",
#         )

#         print(f"Initializing dataset from {self.save_path}.")
#         if Path(self.save_path).is_file():
#             with h5py.File(self.save_path, "r") as hdf:
#                 self.len = hdf[f"{self.data_type}/X"].shape[0]  # Length of the dataset
#         else:
#             print("Please generate preprocessed data using script (preprocess_h5_data.py).")
#             exit()

#     def __len__(self):
#         """Return the length of the current dataset."""
#         return self.len

#     def __getitem__(self, idx):
#         with h5py.File(self.save_path, "r") as hdf:
#             selected = {key: hdf[f"{self.data_type}/{key}"][idx] for key in hdf[f"{self.data_type}"].keys()}

#         # [sequence_len]
#         x = torch.tensor(selected["X"], dtype=torch.int64)

#         x_dict = {}
#         # [sequence_len]
#         x_dict["mode"] = torch.tensor(selected["mode_X"], dtype=torch.int64)
#         # [1]
#         x_dict["user"] = torch.tensor(selected["user_X"][0], dtype=torch.int64)
#         # [sequence_len] in 15 minutes
#         x_dict["time"] = torch.tensor(selected["start_min_X"] // 15, dtype=torch.int64)
#         # [sequence_len]
#         x_dict["length"] = torch.log(torch.tensor(selected["length_X"], dtype=torch.float32))
#         # [sequence_len]
#         x_dict["weekday"] = torch.tensor(selected["weekday_X"], dtype=torch.int64)

#         if self.model_type == "deepmove":
#             x_dict["history_count"] = torch.tensor(selected["history_count"])

#         # Handle padding: if sequences are padded, remove the padding for specific processing
#         # For example, if padding value is 0, and you want to remove trailing padding:
#         if x.size(0) > 0:
#             non_padded_indices = (x != 0).nonzero(as_tuple=True)[0]
#             x = x[non_padded_indices]
#             x_dict["mode"] = x_dict["mode"][non_padded_indices]
#             x_dict["time"] = x_dict["time"][non_padded_indices]
#             x_dict["length"] = x_dict["length"][non_padded_indices]
#             x_dict["weekday"] = x_dict["weekday"][non_padded_indices]

#         y = torch.tensor(selected["loc_Y"], dtype=torch.long)
#         y_mode = torch.tensor(selected["mode_Y"], dtype=torch.long)

#         return x, y, x_dict, y_mode


### Batch caching
class sp_loc_dataset(torch.utils.data.Dataset):
    """
    Dataset class for spatial location prediction.
    Should be used without shuffle in DataLoader.
    Requires preprocessed data in the form of a .h5 file, which is already shuffled.
    """
    def __init__(
        self,
        source_root,
        dataset="gc",
        data_type="train",
        previous_day=7,
        model_type="transformer",
        cache_size=400000,  # Number of samples to cache
    ):
        self.root = source_root
        self.data_type = data_type
        self.previous_day = previous_day
        self.model_type = model_type
        self.dataset = dataset
        self.cache_size = cache_size

        self.data_dir = os.path.join(source_root, "temp")
        self.save_path = os.path.join(
            self.data_dir,
            # f"{self.dataset}_{self.model_type}_{self.previous_day}_preprocessed.h5",
            f"{self.dataset}_{self.model_type}_{self.previous_day}_shuffled.h5",
        )

        print(f"Initializing dataset from {self.save_path}.")
        if Path(self.save_path).is_file():
            with h5py.File(self.save_path, "r") as hdf:
                self.len = hdf[f"{self.data_type}/X"].shape[0]  # Length of the dataset
                print(f"Dataset length ({data_type}): {self.len}")
        else:
            print("Please generate preprocessed data using script (preprocess_h5_data.py) and shuffle it.")
            exit()

        # Initialize cache
        self.cache = {}
        self.cache_start_idx = None

    def __len__(self):
        """Return the length of the current dataset."""
        return self.len

    def _load_cache(self, start_idx):
        print("Get cache")
        """Load a chunk of data into the cache."""
        end_idx = min(start_idx + self.cache_size, self.len)
        with h5py.File(self.save_path, "r") as hdf:
            self.cache = {key: hdf[f"{self.data_type}/{key}"][start_idx:end_idx] for key in hdf[f"{self.data_type}"].keys()}
        self.cache_start_idx = start_idx

    def __getitem__(self, idx):
        # print("Get item")
        if self.cache_start_idx is None or not (self.cache_start_idx <= idx < self.cache_start_idx + self.cache_size):
            # Load the appropriate cache block
            self._load_cache(idx - idx % self.cache_size)

        # Access data from the cache
        cache_idx = idx - self.cache_start_idx
        selected = {key: self.cache[key][cache_idx] for key in self.cache.keys()}
        
        # Assuming the padded sequences have a consistent shape and you want to handle them correctly
        x = torch.tensor(selected["X"], dtype=torch.int64)

        x_dict = {}
        x_dict["mode"] = torch.tensor(selected["mode_X"], dtype=torch.int64)
        x_dict["user"] = torch.tensor(selected["user_X"][0], dtype=torch.int64)
        x_dict["time"] = torch.tensor(selected["start_min_X"] // 15, dtype=torch.int64)
        x_dict["length"] = torch.log(torch.tensor(selected["length_X"], dtype=torch.float32))
        x_dict["weekday"] = torch.tensor(selected["weekday_X"], dtype=torch.int64)

        if self.model_type == "deepmove":
            x_dict["history_count"] = torch.tensor(selected["history_count"])

        # Handle padding: if sequences are padded, remove the padding for specific processing
        if x.size(0) > 0:
            non_padded_indices = (x != 0).nonzero(as_tuple=True)[0]
            x = x[non_padded_indices]
            x_dict["mode"] = x_dict["mode"][non_padded_indices]
            x_dict["time"] = x_dict["time"][non_padded_indices]
            x_dict["length"] = x_dict["length"][non_padded_indices]
            x_dict["weekday"] = x_dict["weekday"][non_padded_indices]

        y = torch.tensor(selected["loc_Y"], dtype=torch.long)
        y_mode = torch.tensor(selected["mode_Y"], dtype=torch.long)

        return x, y, x_dict, y_mode
# import threading
# import torch
# import os
# import h5py
# from pathlib import Path

# class sp_loc_dataset(torch.utils.data.Dataset):
#     """
#     Dataset class for spatial location prediction.
#     Should be used without shuffle in DataLoader.
#     Requires preprocessed data in the form of a .h5 file, which is already shuffled.
#     """
#     def __init__(
#         self,
#         source_root,
#         dataset="gc",
#         data_type="train",
#         previous_day=7,
#         model_type="transformer",
#         cache_size=400000,  # Number of samples to cache
#     ):
#         self.root = source_root
#         self.data_type = data_type
#         self.previous_day = previous_day
#         self.model_type = model_type
#         self.dataset = dataset
#         self.cache_size = cache_size

#         # Construct the path to the preprocessed data file
#         self.data_dir = os.path.join(source_root, "temp")
#         self.save_path = os.path.join(
#             self.data_dir,
#             f"{self.dataset}_{self.model_type}_{self.previous_day}_shuffled.h5",
#         )

#         # Check if the data file exists and get the length of the dataset
#         if Path(self.save_path).is_file():
#             with h5py.File(self.save_path, "r") as hdf:
#                 self.len = hdf[f"{self.data_type}/X"].shape[0]
#         else:
#             exit()

#         # Initialize the cache and threading-related attributes
#         self.cache = {}
#         self.cache_start_idx = None
#         self.next_cache = {}
#         self.next_cache_start_idx = None
#         self.preload_thread = None
#         self.lock = threading.Lock()

#     def __len__(self):
#         """Return the length of the current dataset."""
#         return self.len

#     def _load_cache(self, start_idx):
#         """
#         Load a chunk of data into the cache starting from a specific index.
#         This method also initiates the preloading of the next chunk.
#         """
#         print("Loading cache")
#         end_idx = min(start_idx + self.cache_size, self.len)
#         with h5py.File(self.save_path, "r") as hdf:
#             with self.lock:
#                 self.cache = {key: hdf[f"{self.data_type}/{key}"][start_idx:end_idx] for key in hdf[f"{self.data_type}"].keys()}
#                 self.cache_start_idx = start_idx

#         # Preload the next chunk if there is more data left
#         if end_idx < self.len:
#             self._preload_next_cache(end_idx)

#     def _preload_next_cache(self, start_idx):
#         """
#         Preload the next cache chunk in a separate thread to avoid waiting during data access.
#         """
#         if self.preload_thread and self.preload_thread.is_alive():
#             return  # Skip preloading if the previous preload is still running

#         def preload():
#             print(f"Starting to preload cache starting at index {start_idx}.")
#             end_idx = min(start_idx + self.cache_size, self.len)
#             with h5py.File(self.save_path, "r") as hdf:
#                 with self.lock:
#                     self.next_cache = {key: hdf[f"{self.data_type}/{key}"][start_idx:end_idx] for key in hdf[f"{self.data_type}"].keys()}
#                     self.next_cache_start_idx = start_idx
#             print(f"Finished preloading cache from index {start_idx} to {end_idx}.")

#         self.preload_thread = threading.Thread(target=preload)
#         self.preload_thread.start()

#     def _switch_to_next_cache(self):
#         """
#         Switch to the preloaded cache chunk.
#         This method is called when the data being accessed falls within the preloaded chunk.
#         """
#         print("\n Switching to next cache")
#         with self.lock:
#             self.cache = self.next_cache
#             self.cache_start_idx = self.next_cache_start_idx
#             self.next_cache = {}
#             self.next_cache_start_idx = None
        

#     def __getitem__(self, idx):
#         """
#         Retrieve a data sample from the dataset at the specified index.
#         This method handles cache loading and switching.
#         """
#         if self.cache_start_idx is None or not (self.cache_start_idx <= idx < self.cache_start_idx + self.cache_size):
#             # If the requested index is not within the current cache, load the appropriate cache block
#             if self.next_cache_start_idx is not None and self.next_cache_start_idx <= idx < self.next_cache_start_idx + self.cache_size:
#                 self._switch_to_next_cache()
#                 # Start preloading the next block immediately after switching
#                 next_start_idx = self.cache_start_idx + self.cache_size
#                 if next_start_idx < self.len:
#                     self._preload_next_cache(next_start_idx)
#             else:
#                 self._load_cache(idx - idx % self.cache_size)

#         # Access data from the cache
#         cache_idx = idx - self.cache_start_idx
#         selected = {key: self.cache[key][cache_idx] for key in self.cache.keys()}
        
#         # Convert the data to tensors
#         x = torch.tensor(selected["X"], dtype=torch.int64)

#         x_dict = {}
#         x_dict["mode"] = torch.tensor(selected["mode_X"], dtype=torch.int64)
#         x_dict["user"] = torch.tensor(selected["user_X"][0], dtype=torch.int64)
#         x_dict["time"] = torch.tensor(selected["start_min_X"] // 15, dtype=torch.int64)
#         x_dict["length"] = torch.log(torch.tensor(selected["length_X"], dtype=torch.float32))
#         x_dict["weekday"] = torch.tensor(selected["weekday_X"], dtype=torch.int64)

#         if self.model_type == "deepmove":
#             x_dict["history_count"] = torch.tensor(selected["history_count"])

#         # Handle padding: remove padding if sequences are padded
#         if x.size(0) > 0:
#             non_padded_indices = (x != 0).nonzero(as_tuple=True)[0]
#             x = x[non_padded_indices]
#             x_dict["mode"] = x_dict["mode"][non_padded_indices]
#             x_dict["time"] = x_dict["time"][non_padded_indices]
#             x_dict["length"] = x_dict["length"][non_padded_indices]
#             x_dict["weekday"] = x_dict["weekday"][non_padded_indices]

#         y = torch.tensor(selected["loc_Y"], dtype=torch.long)
#         y_mode = torch.tensor(selected["mode_Y"], dtype=torch.long)

#         #Check for any Nan causing issues
#         if torch.isnan(x).any() or torch.isnan(y).any() or torch.isnan(y_mode).any():
#             print("Nan detected")
#             print(x)
#             print(x_dict)
#             print(y)
#             print(y_mode)
#             exit()

#         return x, y, x_dict, y_mode



    def generate_data(self):
        save_path = os.path.join(
            self.data_dir,
            f"{self.dataset}_{self.model_type}_{self.previous_day}_{self.data_type}.h5",
        )

        with h5py.File(save_path, "w") as hdf:
            # Process location data if model_type is "mobtcast"
            if self.model_type == "mobtcast":
                loc_file = h5py.File(os.path.join(self.root, f"locations_{self.dataset}.h5"), "r")
                hdf.create_dataset("locations/lng", data=loc_file["lng"][:], compression="gzip", compression_opts=9)
                hdf.create_dataset("locations/lat", data=loc_file["lat"][:], compression="gzip", compression_opts=9)
                print("Processed and saved location data.")

            ori_data = h5py.File(os.path.join(self.root, f"dataSet_{self.dataset}.h5"), "r")

            # Assuming chunk size for processing
            chunk_size = 10000
            num_rows = ori_data["user_id"].shape[0]

            all_records = []

            for start in range(0, num_rows, chunk_size):
                end = min(start + chunk_size, num_rows)
                chunk = {key: ori_data[key][start:end] for key in ori_data.keys()}

                # Convert chunk to DataFrame for easier processing
                chunk_df = pd.DataFrame(chunk)
                chunk_df.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)

                # Encode mode
                if start == 0:  # Fit encoder on the first chunk
                    enc_mode = OrdinalEncoder(dtype=np.int64).fit(chunk_df["mode"].values.reshape(-1, 1))
                chunk_df["mode"] = enc_mode.transform(chunk_df["mode"].values.reshape(-1, 1)) + 1

                # Truncate duration
                chunk_df.loc[chunk_df["duration"] > 60 * 24 * 2 - 1, "duration"] = 60 * 24 * 2 - 1

                # Split chunk into train, validation, test
                train_data, vali_data, test_data = self.splitDataset(chunk_df)

                # Encode user_id and location_id
                if start == 0:  # Fit encoder on the first chunk
                    enc_user = OrdinalEncoder(dtype=np.int64).fit(train_data["user_id"].values.reshape(-1, 1))
                    enc_loc = OrdinalEncoder(
                        dtype=np.int64,
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ).fit(train_data["location_id"].values.reshape(-1, 1))

                for data in [train_data, vali_data, test_data]:
                    data["user_id"] = enc_user.transform(data["user_id"].values.reshape(-1, 1)) + 1
                    data["location_id"] = enc_loc.transform(data["location_id"].values.reshape(-1, 1)) + 2

                # Process datasets
                for dataset_type, data in zip(["train", "validation", "test"], [train_data, vali_data, test_data]):
                    valid_records = self.preProcessDatasets(data, dataset_type)
                    all_records.extend(valid_records)

                    for key in valid_records[0].keys():
                        data_to_save = np.array([record[key] for record in valid_records])

                        if f"{dataset_type}/{key}" not in hdf:
                            hdf.create_dataset(
                                f"{dataset_type}/{key}",
                                data=data_to_save,
                                maxshape=(None,),
                                chunks=True,
                                compression="gzip",
                                compression_opts=9
                            )
                        else:
                            dataset = hdf[f"{dataset_type}/{key}"]
                            dataset.resize((dataset.shape[0] + data_to_save.shape[0]), axis=0)
                            dataset[-data_to_save.shape[0]:] = data_to_save

                print(f"Processed and saved {dataset_type} dataset chunk: {start} to {end}.")

        # Return the appropriate records based on data_type
        if self.data_type == "test":
            return [record for record in all_records if record['type'] == 'test']
        elif self.data_type == "validation":
            return [record for record in all_records if record['type'] == 'validation']
        elif self.data_type == "train":
            return [record for record in all_records if record['type'] == 'train']


    def splitDataset(self, totalData):
        """Split dataset into train, vali and test."""
        totalData = totalData.groupby("user_id").apply(self.getSplitDaysUser)

        train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
        vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
        test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

        # final cleaning
        train_data.drop(columns={"Dataset"}, inplace=True)
        vali_data.drop(columns={"Dataset"}, inplace=True)
        test_data.drop(columns={"Dataset"}, inplace=True)

        train_u = set(train_data["user_id"].unique())
        val_u = set(vali_data["user_id"].unique())
        test_u = set(test_data["user_id"].unique())
        u = set.intersection(train_u, val_u, test_u)
        train_data = train_data.loc[train_data["user_id"].isin(u)]
        vali_data = vali_data.loc[vali_data["user_id"].isin(u)]
        test_data = test_data.loc[test_data["user_id"].isin(u)]
        #Drop the 'user_id' index column
        train_data.reset_index(drop=True, inplace=True)
        vali_data.reset_index(drop=True, inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        return train_data, vali_data, test_data

    def getSplitDaysUser(self, df):
        """Split the dataset according to the tracked day of each user."""
        maxDay = df["start_day"].max()
        train_split = maxDay * 0.6
        vali_split = maxDay * 0.8

        df["Dataset"] = "test"
        df.loc[df["start_day"] < train_split, "Dataset"] = "train"
        df.loc[
            (df["start_day"] >= train_split) & (df["start_day"] < vali_split),
            "Dataset",
        ] = "vali"
        # counts = df["Dataset"].value_counts()
        # if len(counts) != 3:
        #     # Need this for texas dataset
        #     #Split the dataset into train, vali and test by the quantile number of entries: 60%, 20%, 20%
        #     sub_df = df[['timestamp']].astype(int)
        #     train_split = sub_df['timestamp'].quantile(0.6)
        #     vali_split = sub_df['timestamp'].quantile(0.8)
        #     sub_df["Dataset"] = "test"
        #     sub_df.loc[sub_df["timestamp"] < train_split, "Dataset"] = "train"
        #     sub_df.loc[
        #         (sub_df["timestamp"] >= train_split) & (sub_df["timestamp"] < vali_split),
        #         "Dataset",
        #     ] = "vali"

        #     df["Dataset"] = sub_df["Dataset"]
        
        # counts = df["Dataset"].value_counts()
        # if len(counts) != 3:
        #     print("Error in spliting the dataset")
        #     print(counts)
        #     print(df)
        #     exit()
        return df

    def preProcessDatasets(self, data, dataset_type):
        """Generate the datasets and save to the disk."""
        valid_records = self.getValidSequence(data)

        save_path = os.path.join(
            self.data_dir,
            f"{self.dataset}_{self.model_type}_{self.previous_day}_{dataset_type}.pk",
        )
        save_pk_file(save_path, valid_records)

        return valid_records

    def getValidSequence(self, input_df):
        valid_user_ls = applyParallel(input_df.groupby("user_id"), self.getValidSequenceUser, n_jobs=-1)
        return [item for sublist in valid_user_ls for item in sublist]

    def getValidSequenceUser(self, df):

        df.reset_index(drop=True, inplace=True)

        data_single_user = []
        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days

        for index, row in df.iterrows():
            # exclude the first records
            if row["diff_day"] < self.previous_day:
                continue

            hist = df.iloc[:index]
            hist = hist.loc[(hist["start_day"] >= (row["start_day"] - self.previous_day))]

            # should be at least contain 2 history locations
            if len(hist) < 3:
                continue

            data_dict = {}
            # only for deepmove: consider the last 2 days as curr and remaining 5 days as history
            if self.model_type == "deepmove":
                # and all other as history
                data_dict["history_count"] = len(hist.loc[hist["start_day"] < (row["start_day"] - 1)])
                # the history sequence and the current sequence shall not be 0
                if data_dict["history_count"] == 0 or data_dict["history_count"] == len(hist):
                    continue

            data_dict["X"] = hist["location_id"].values
            data_dict["user_X"] = hist["user_id"].values
            data_dict["start_min_X"] = hist["start_min"].values
            data_dict["mode_X"] = hist["mode"].values
            data_dict["length_X"] = hist["length_m"].values
            data_dict["weekday_X"] = hist["weekday"].values

            # the next location is the Y
            data_dict["loc_Y"] = int(row["location_id"])
            # the next mode is the mode_Y
            data_dict["mode_Y"] = int(row["mode"])

            data_single_user.append(data_dict)

        return data_single_user


def applyParallel(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    """
    Funtion warpper to parallelize funtions after .groupby().
    Parameters
    ----------
    dfGrouped: pd.DataFrameGroupBy
        The groupby object after calling df.groupby(COLUMN).
    func: function
        Function to apply to the dfGrouped object, i.e., dfGrouped.apply(func).
    n_jobs: int
        The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging. See
        https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
        for a detailed description
    print_progress: boolean
        If set to True print the progress of apply.
    **kwargs:
        Other arguments passed to func.
    Returns
    -------
    pd.DataFrame:
        The result of dfGrouped.apply(func)
    """
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return df_ls


def save_pk_file(save_path, data):
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pk_file(save_path):
    return pickle.load(open(save_path, "rb"))


def collate_fn(batch):
    """function to collate data samples into batch tensors."""
    x_batch, y_batch, y_mode_batch = [], [], []

    x_dict_batch = {"len": []}
    for key in batch[0][-2]:
        x_dict_batch[key] = []

    for x_sample, y_sample, x_dict_sample, y_mode_sample in batch:
        x_batch.append(x_sample)
        y_batch.append(y_sample)
        y_mode_batch.append(y_mode_sample)

        # x_dict_sample
        x_dict_batch["len"].append(len(x_sample))
        for key in x_dict_sample:
            x_dict_batch[key].append(x_dict_sample[key])

    x_batch = pad_sequence(x_batch)
    y_batch = torch.tensor(y_batch, dtype=torch.int64)
    y_mode_batch = torch.tensor(y_mode_batch, dtype=torch.int64)

    # x_dict_batch
    x_dict_batch["user"] = torch.tensor(x_dict_batch["user"], dtype=torch.int64)
    x_dict_batch["len"] = torch.tensor(x_dict_batch["len"], dtype=torch.int64)
    for key in x_dict_batch:
        if key in ["user", "len", "history_count"]:
            continue
        x_dict_batch[key] = pad_sequence(x_dict_batch[key])

    return x_batch, y_batch, x_dict_batch, y_mode_batch



def deepmove_collate_fn(batch):
    """function to collate data samples into batch tensors."""
    # history_batch, curr_batch, tgt_batch = [], [], []
    x_batch, hist_batch, y_batch, y_mode_batch = [], [], [], []

    # the context
    x_dict_batch = {"len": []}
    for key in batch[0][-2]:
        if key in ["history_count"]:
            continue
        x_dict_batch[key] = []
    hist_dict_batch = {"len": []}
    for key in batch[0][-2]:
        if key in ["history_count"]:
            continue
        hist_dict_batch[key] = []

    # x_sample, y_sample, x_dict_sample, y_mode_sample
    for x_sample, y_sample, x_dict_sample, y_mode_sample in batch:
        history_len = x_dict_sample["history_count"]

        hist_batch.append(x_sample[:history_len])
        x_batch.append(x_sample[history_len:])
        y_batch.append(y_sample)
        y_mode_batch.append(y_mode_sample)

        # hist_dict_batch
        hist_dict_batch["len"].append(history_len)
        hist_dict_batch["user"].append(x_dict_sample["user"])
        for key in x_dict_sample:
            if key in ["user", "history_count"]:
                continue
            hist_dict_batch[key].append(x_dict_sample[key][:history_len])

        # x_dict_batch
        x_dict_batch["len"].append(len(x_sample[history_len:]))
        x_dict_batch["user"].append(x_dict_sample["user"])
        for key in x_dict_sample:
            if key in ["user", "history_count"]:
                continue
            x_dict_batch[key].append(x_dict_sample[key][history_len:])

    x_batch = pad_sequence(x_batch)
    hist_batch = pad_sequence(hist_batch)
    y_batch = torch.tensor(y_batch, dtype=torch.int64)
    y_mode_batch = torch.tensor(y_mode_batch, dtype=torch.int64)

    # hist_dict_batch
    hist_dict_batch["user"] = torch.tensor(hist_dict_batch["user"], dtype=torch.int64)
    hist_dict_batch["len"] = torch.tensor(hist_dict_batch["len"], dtype=torch.int64)
    for key in hist_dict_batch:
        if key in ["user", "len"]:
            continue
        hist_dict_batch[key] = pad_sequence(hist_dict_batch[key])

    # x_dict_batch
    x_dict_batch["user"] = torch.tensor(x_dict_batch["user"], dtype=torch.int64)
    x_dict_batch["len"] = torch.tensor(x_dict_batch["len"], dtype=torch.int64)
    for key in x_dict_batch:
        if key in ["user", "len"]:
            continue
        x_dict_batch[key] = pad_sequence(x_dict_batch[key])

    return ((hist_batch, x_batch), y_batch, (hist_dict_batch, x_dict_batch), y_mode_batch)


def test_dataloader(train_loader):
    ave_shape = 0
    hist_shape = 0
    y_shape = 0
    y_mode_shape = 0
    user_ls = []
    for batch_idx, (x, y, x_dict, y_mode) in tqdm(enumerate(train_loader)):
        # print("batch_idx ", batch_idx)
        # print(inputs.shape)
        (hist, x) = x
        (hist_dict, x_dict) = x_dict

        hist_shape += hist.shape[0]
        ave_shape += x.shape[0]
        y_shape += y.shape[0]
        y_mode_shape += y_mode.shape[0]

        user_ls.extend(x_dict["user"])

    # print(np.max(user_ls), np.min(user_ls))
    print(hist_shape / len(train_loader))
    print(ave_shape / len(train_loader))
    print(y_mode_shape / len(train_loader))
    print(y_shape / len(train_loader))


if __name__ == "__main__":
    source_root = r"./data/"

    dataset_train = sp_loc_dataset(
        source_root, data_type="train", dataset="geolife", previous_day=7, model_type="transformer"
    )
    dataset_val = sp_loc_dataset(
        source_root, data_type="validation", dataset="geolife", previous_day=7, model_type="transformer"
    )
    dataset_test = sp_loc_dataset(
        source_root, data_type="test", dataset="geolife", previous_day=7, model_type="transformer"
    )

    kwds = {"shuffle": False, "num_workers": 0, "batch_size": 2}
    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, **kwds)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=collate_fn, **kwds)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=collate_fn, **kwds)

    test_dataloader(train_loader)
    test_dataloader(val_loader)
    test_dataloader(test_loader)
