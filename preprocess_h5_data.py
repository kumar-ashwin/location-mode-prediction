import h5py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from torch.nn.utils.rnn import pad_sequence
import torch
from datetime import datetime, timedelta
import os
from joblib import Parallel, delayed
from tqdm import tqdm

# Contains the preprocessing functions for converting into trajcectory data and saving to HDF5
# Also contains the function to shuffle the data in an existing HDF5 file

def preprocess_and_save_dataset(source_root, dataset_name, output_path, chunk_size=10000, previous_days=7):
    # Initialize sets to collect unique categories
    unique_user_ids = set()
    unique_location_ids = set()
    unique_modes = set()
    max_sequence_length = 0  # Track maximum sequence length

    # First pass: Collect unique categories across the entire dataset
    ori_data = h5py.File(os.path.join(source_root, f"dataSet_{dataset_name}.h5"), "r")
    num_rows = ori_data["user_id"].shape[0]

    user_id_data = ori_data["user_id"]

    start = 0
    while start < num_rows:
        end = min(start + chunk_size, num_rows)

        # Ensure the chunk includes the full data for the last user in this chunk
        while end < num_rows and user_id_data[end - 1] == user_id_data[end]:
            end += 1

        chunk = {key: ori_data[key][start:end] for key in ori_data.keys()}
        chunk_df = pd.DataFrame(chunk)

        unique_user_ids.update(chunk_df["user_id"].unique())
        unique_location_ids.update(chunk_df["location_id"].unique())
        unique_modes.update(chunk_df["mode"].unique())

        max_sequence_length = max(max_sequence_length, chunk_df.groupby("user_id").size().max())

        start = end

    print("Max sequence length:", max_sequence_length)
    print("Number of unique users:", len(unique_user_ids))
    print("Number of unique locations:", len(unique_location_ids))
    print("Number of unique modes:", len(unique_modes))
    print('Total rows:', num_rows)

    unique_user_ids = sorted(list(unique_user_ids))
    unique_location_ids = sorted(list(unique_location_ids))
    unique_modes = sorted(list(unique_modes))

    # Fit the encoders on the complete set of unique categories
    enc_user = OrdinalEncoder(categories=[unique_user_ids], dtype=np.int64)
    enc_loc = OrdinalEncoder(categories=[unique_location_ids], dtype=np.int64)
    enc_mode = OrdinalEncoder(categories=[unique_modes], dtype=np.int64)

    # Manually fit the encoders
    enc_user.fit(np.array(unique_user_ids).reshape(-1, 1))
    enc_loc.fit(np.array(unique_location_ids).reshape(-1, 1))
    enc_mode.fit(np.array(unique_modes).reshape(-1, 1))

    # Second pass: Apply encoding, split data, and save to HDF5
    start = 0
    with h5py.File(output_path, "w") as hdf:
        while start < num_rows:
            end = min(start + chunk_size, num_rows)

            # Extend end to ensure the chunk includes the full data for the last user in this chunk
            while end < num_rows and user_id_data[end - 1] == user_id_data[end]:
                end += 1

            chunk = {key: ori_data[key][start:end] for key in ori_data.keys()}
            chunk_df = pd.DataFrame(chunk)
            chunk_df['timestamp'] = pd.to_datetime(chunk_df['timestamp'], unit='s')
            chunk_df.sort_values(by=["user_id", "timestamp"], inplace=True)

            # Apply encodings. Offset by 1 to reserve 0 for padding
            chunk_df["user_id"] = enc_user.transform(chunk_df["user_id"].values.reshape(-1, 1)) + 1
            chunk_df["location_id"] = enc_loc.transform(chunk_df["location_id"].values.reshape(-1, 1)) + 2
            chunk_df["mode"] = enc_mode.transform(chunk_df["mode"].values.reshape(-1, 1)) + 1

            # Split data based on custom date ranges
            train_data, vali_data, test_data = split_by_date(chunk_df, previous_days)

            # Process and save datasets
            for dataset_type, data in zip(["train", "validation", "test"], [train_data, vali_data, test_data]):
                valid_records = get_valid_sequences(data, previous_days)

                if valid_records:    
                    for key in valid_records[0].keys():
                        if isinstance(valid_records[0][key], np.ndarray):
                            # If the field is a sequence (i.e., an array)
                            sequences = [torch.tensor(record[key]) for record in valid_records]
                            if sequences:
                                padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0).numpy()
                                # Pad sequences to max length
                                if padded_sequences.shape[1] < max_sequence_length:
                                    padded_sequences = np.pad(padded_sequences, ((0, 0), (0, max_sequence_length - padded_sequences.shape[1])), mode='constant')

                                if f"{dataset_type}/{key}" not in hdf:
                                    hdf.create_dataset(
                                        f"{dataset_type}/{key}",
                                        data=padded_sequences,
                                        maxshape=(None, max_sequence_length),
                                        chunks=True,
                                        compression="gzip",
                                        compression_opts=9
                                    )
                                else:
                                    dataset = hdf[f"{dataset_type}/{key}"]
                                    dataset.resize((dataset.shape[0] + padded_sequences.shape[0]), axis=0)
                                    dataset[-padded_sequences.shape[0]:] = padded_sequences
                        else:
                            # If the field is a single value (i.e., an int)
                            values = np.array([record[key] for record in valid_records])
                            if f"{dataset_type}/{key}" not in hdf:
                                hdf.create_dataset(
                                    f"{dataset_type}/{key}",
                                    data=values,
                                    maxshape=(None,),
                                    chunks=True,
                                    compression="gzip",
                                    compression_opts=9
                                )
                            else:
                                dataset = hdf[f"{dataset_type}/{key}"]
                                dataset.resize((dataset.shape[0] + values.shape[0]), axis=0)
                                dataset[-values.shape[0]:] = values

            print(f"Processed and saved chunk: {start} to {end}.")
            start = end 

def split_by_date(df, previous_days):
    """
    Splits the dataset into train, validation, and test based on date ranges.
    Retains the previous n days of context for validation and test sets.
    """
    train_start = datetime(2021, 1, 1)
    train_end = datetime(2021, 3, 1)
    val_start = datetime(2021, 3, 1)
    val_end = datetime(2021, 3, 16)
    test_start = datetime(2021, 3, 16)

    # Initialize empty DataFrames for each split
    train_data = pd.DataFrame()
    vali_data = pd.DataFrame()
    test_data = pd.DataFrame()

    # Group by user to retain context across splits
    for user_id, user_data in df.groupby("user_id"):
        user_data = user_data.sort_values("timestamp")

        train_user = user_data[(user_data["timestamp"] >= train_start) & (user_data["timestamp"] < train_end)]
        val_user = user_data[(user_data["timestamp"] >= val_start) & (user_data["timestamp"] < val_end)]
        test_user = user_data[user_data["timestamp"] >= test_start]

        # Adjust validation and test sets to include the previous n days
        if not val_user.empty:
            first_val_date = val_user["timestamp"].min()
            context_start = first_val_date - timedelta(days=previous_days)
            val_user = pd.concat([user_data[(user_data["timestamp"] >= context_start) & (user_data["timestamp"] < first_val_date)], val_user])

        if not test_user.empty:
            first_test_date = test_user["timestamp"].min()
            context_start = first_test_date - timedelta(days=previous_days)
            test_user = pd.concat([user_data[(user_data["timestamp"] >= context_start) & (user_data["timestamp"] < first_test_date)], test_user])

        train_data = pd.concat([train_data, train_user])
        vali_data = pd.concat([vali_data, val_user])
        test_data = pd.concat([test_data, test_user])

    return train_data, vali_data, test_data

def get_valid_sequences(input_df, previous_days):
    valid_user_ls = applyParallel(input_df.groupby("user_id"), getValidSequenceUser, n_jobs=-1, previous_days=previous_days)
    return [item for sublist in valid_user_ls for item in sublist]

def getValidSequenceUser(df, previous_days):

    df.reset_index(drop=True, inplace=True)

    data_single_user = []
    min_days = df["start_day"].min()
    df["diff_day"] = df["start_day"] - min_days

    for index, row in df.iterrows():
        # exclude the first records
        if row["diff_day"] < previous_days:
            continue

        hist = df.iloc[:index]
        hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_days))]

        # should be at least contain 2 history locations
        if len(hist) < 3:
            continue

        data_dict = {}
        # only for deepmove: consider the last 2 days as curr and remaining 5 days as history
        # data_dict["history_count"] = len(hist.loc[hist["start_day"] < (row["start_day"] - 1)])
        # if data_dict["history_count"] == 0 or data_dict["history_count"] == len(hist):
        #     continue

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
    Function wrapper to parallelize functions after .groupby().
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

# def shuffle_hdf5_groups(input_path, output_path, chunk_size=10000):
#     # Define the groups that you want to shuffle
#     group_names = ["train", "validation", "test"]

#     # Open the original HDF5 file
#     with h5py.File(input_path, "r") as hdf_in:
#         # Open a new HDF5 file for the shuffled data
#         with h5py.File(output_path, "w") as hdf_out:
#             for group_name in group_names:
#                 group = hdf_in[group_name]
                
#                 # Assume all datasets in the group have the same number of samples (you should validate this if necessary)
#                 first_dataset = list(group.keys())[0]
#                 num_samples = group[first_dataset].shape[0]

#                 # Create an array of indices and shuffle it
#                 indices = np.arange(num_samples)
#                 np.random.shuffle(indices)

#                 # Create a new group in the output file
#                 out_group = hdf_out.create_group(group_name)

#                 # Shuffle each dataset within the group
#                 for key in group.keys():
#                     original_dataset = group[key]
#                     # Create corresponding dataset in the output file
#                     out_dataset = out_group.create_dataset(key, shape=original_dataset.shape, dtype=original_dataset.dtype, maxshape=original_dataset.shape, chunks=True)

#                     # Shuffle data by writing in the order of shuffled indices
#                     for start in range(0, num_samples, chunk_size):
#                         end = min(start + chunk_size, num_samples)
#                         chunk_indices = indices[start:end]

#                         # Sort indices to ensure they are in increasing order
#                         sorted_chunk_indices = np.sort(chunk_indices)
#                         sorted_data_chunk = original_dataset[sorted_chunk_indices]

#                         # Reorder data chunk to match the shuffled order
#                         original_order = np.argsort(chunk_indices)
#                         shuffled_data_chunk = sorted_data_chunk[original_order]

#                         out_dataset[start:end] = shuffled_data_chunk

#                     print(f"Shuffled dataset '{key}' in group '{group_name}' from {start} to {end}")

#     print(f"Shuffling complete. Shuffled data saved to {output_path}")


# import h5py
# import numpy as np

# def shuffle_hdf5_groups(input_path, output_path, chunk_size=10000, compression="gzip", compression_opts=9):
#     # Define the groups that you want to shuffle
#     group_names = ["train", "validation", "test"]

#     # Open the original HDF5 file
#     with h5py.File(input_path, "r") as hdf_in:
#         # Open a new HDF5 file for the shuffled data
#         with h5py.File(output_path, "w") as hdf_out:
#             for group_name in group_names:
#                 group = hdf_in[group_name]

#                 # Assume all datasets in the group have the same number of samples
#                 first_dataset = list(group.keys())[0]
#                 num_samples = group[first_dataset].shape[0]

#                 # Create an array of indices and shuffle it
#                 indices = np.arange(num_samples)
#                 np.random.shuffle(indices)

#                 # Shuffle each dataset within the group
#                 for key in group.keys():
#                     original_dataset = group[key]
                    
#                     # Create corresponding dataset in the output file with the same properties as the original
#                     out_dataset = hdf_out.create_dataset(
#                         f"{group_name}/{key}",
#                         shape=original_dataset.shape,
#                         dtype=original_dataset.dtype,
#                         maxshape=original_dataset.maxshape,  # Preserve maxshape from the original
#                         chunks=original_dataset.chunks,  # Preserve chunking from the original
#                         compression=compression,  # Preserve compression from the original
#                         compression_opts=compression_opts
#                     )

#                     # Shuffle and write the data in chunks
#                     for start in range(0, num_samples, chunk_size):
#                         end = min(start + chunk_size, num_samples)
#                         chunk_indices = indices[start:end]

#                         # Access data using sorted indices
#                         sorted_chunk_indices = np.sort(chunk_indices)
#                         sorted_data_chunk = original_dataset[sorted_chunk_indices]

#                         # Reorder the sorted data chunk according to the original shuffled indices
#                         original_order = np.argsort(chunk_indices)
#                         shuffled_data_chunk = sorted_data_chunk[original_order]

#                         # Write the shuffled chunk to the output dataset
#                         out_dataset[start:end] = shuffled_data_chunk

#                     print(f"Shuffled dataset '{key}' in group '{group_name}'.")

#     print(f"Shuffling complete. Shuffled data saved to {output_path}")

# import h5py
# import numpy as np

# def shuffle_hdf5_groups(input_path, output_path, chunk_size=100000):
#     # Define the groups that you want to shuffle
#     group_names = ["train", "validation", "test"]

#     # Open the original HDF5 file
#     with h5py.File(input_path, "r") as hdf_in:
#         # Open a new HDF5 file for the shuffled data
#         with h5py.File(output_path, "w") as hdf_out:
#             for group_name in group_names:
#                 group = hdf_in[group_name]

#                 # Assume all datasets in the group have the same number of samples
#                 first_dataset = list(group.keys())[0]
#                 num_samples = group[first_dataset].shape[0]

#                 # Create an array of indices and shuffle it
#                 indices = np.arange(num_samples)
#                 np.random.shuffle(indices)

#                 # Process each dataset within the group
#                 for key in group.keys():
#                     original_dataset = group[key]

#                     # Create corresponding dataset in the output file with the same properties as the original
#                     out_dataset = hdf_out.create_dataset(
#                         f"{group_name}/{key}",
#                         shape=original_dataset.shape,
#                         dtype=original_dataset.dtype,
#                         chunks=original_dataset.chunks,
#                         compression=original_dataset.compression,
#                         compression_opts=original_dataset.compression_opts
#                     )

#                     # Shuffle and write the data in chunks
#                     for start in range(0, num_samples, chunk_size):
#                         end = min(start + chunk_size, num_samples)
#                         chunk_indices = indices[start:end]

#                         # Sort indices for proper HDF5 access and shuffle within each chunk
#                         sorted_chunk_indices = np.sort(chunk_indices)
#                         sorted_data_chunk = original_dataset[sorted_chunk_indices]
#                         original_order = np.argsort(chunk_indices)
#                         shuffled_data_chunk = sorted_data_chunk[original_order]

#                         # Write the shuffled chunk to the output dataset
#                         out_dataset[start:end] = shuffled_data_chunk

#                     print(f"Shuffled dataset '{key}' in group '{group_name}'.")

#     print(f"Shuffling complete. Shuffled data saved to {output_path}")


import h5py
import numpy as np

def shuffle_hdf5_groups(input_path, output_path, chunk_size=10000):
    # Fast version that also doesnt explore the dataset size
    group_names = ["train", "validation", "test"]

    # Open the original HDF5 file
    with h5py.File(input_path, "r") as hdf_in:
        # Open a new HDF5 file for the shuffled data
        with h5py.File(output_path, "w") as hdf_out:
            for group_name in group_names:
                group = hdf_in[group_name]
                
                # Assume all datasets in the group have the same number of samples
                first_dataset = list(group.keys())[0]
                num_samples = group[first_dataset].shape[0]

                # Create an array of indices and shuffle it
                indices = np.arange(num_samples)
                np.random.shuffle(indices)

                # Create a new group in the output file
                out_group = hdf_out.create_group(group_name)

                # Shuffle each dataset within the group
                for key in group.keys():
                    original_dataset = group[key]
                    
                    # Determine chunk size and compression (if needed)
                    chunk_shape = (min(chunk_size, num_samples),) + original_dataset.shape[1:]
                    out_dataset = out_group.create_dataset(
                        key, 
                        shape=original_dataset.shape, 
                        dtype=original_dataset.dtype, 
                        maxshape=original_dataset.shape, 
                        chunks=chunk_shape,
                        compression="gzip",  # Adjust or remove compression as necessary
                        compression_opts=4   # Adjust compression level
                    )

                    # Shuffle data by writing in the order of shuffled indices
                    print(f"Shuffling dataset '{key}' in group '{group_name}'...")
                    print(f"Chunk size: {chunk_size}, Total samples: {num_samples}, Total chunks: {num_samples // chunk_size + 1}")
                    iter = 0
                    for start in range(0, num_samples, chunk_size):
                        iter += 1
                        print(f"Chunk {iter} of {num_samples // chunk_size + 1}")
                        end = min(start + chunk_size, num_samples)
                        chunk_indices = indices[start:end]

                        # Sort indices to ensure they are in increasing order
                        print("Sorting indices...")
                        sorted_chunk_indices = np.sort(chunk_indices)
                        print("Accessing data chunk...")
                        sorted_data_chunk = original_dataset[sorted_chunk_indices]

                        # Reorder data chunk to match the shuffled order
                        print("Reordering data chunk...")
                        original_order = np.argsort(chunk_indices)
                        shuffled_data_chunk = sorted_data_chunk[original_order]

                        # Write the shuffled chunk to the output dataset
                        print("Writing shuffled data...")
                        print(f"Writing from {start} to {end}")
                        out_dataset[start:end] = shuffled_data_chunk

                    print(f"Shuffled dataset '{key}' in group '{group_name}' from {start} to {end}")

    print(f"Shuffling complete. Shuffled data saved to {output_path}")

import numpy as np
import h5py

def shuffle_hdf5_groups_efficient(input_path, output_path, chunk_size=400000):
    group_names = ["train", "validation", "test"]

    # Open the original HDF5 file
    with h5py.File(input_path, "r") as hdf_in:
        # Open a new HDF5 file for the shuffled data
        with h5py.File(output_path, "w") as hdf_out:
            for group_name in group_names:
                group = hdf_in[group_name]

                # Assume all datasets in the group have the same number of samples
                first_dataset = list(group.keys())[0]
                num_samples = group[first_dataset].shape[0]

                # Create a new group in the output file
                out_group = hdf_out.create_group(group_name)

                print(f"Shuffling group '{group_name}' with {num_samples} samples...")

                # Process the data in chunks
                for start in range(0, num_samples, chunk_size):
                    end = min(start + chunk_size, num_samples)
                    chunk_length = end - start

                    # Generate a shuffle index for this chunk
                    shuffle_indices = np.arange(chunk_length)
                    np.random.shuffle(shuffle_indices)

                    # Shuffle each dataset within the group using the chunk-specific shuffle index
                    for key in group.keys():
                        print(f"key: {key}, start: {start}, end: {end}, group_name: {group_name}")
                        original_dataset = group[key]

                        # Create the output dataset with the same shape and dtype if not already created
                        if key not in out_group:
                            out_dataset = out_group.create_dataset(
                                key,
                                shape=original_dataset.shape,
                                dtype=original_dataset.dtype,
                                chunks=(min(chunk_size, num_samples),) + original_dataset.shape[1:],
                                compression="gzip",  # Optional: adjust or remove compression as necessary
                                compression_opts=4   # Optional: adjust compression level
                            )
                        else:
                            out_dataset = out_group[key]

                        # Load data for the chunk in its original order
                        # print(f"Loading chunk from {start} to {end}...")
                        data_chunk = original_dataset[start:end]

                        # Apply the shuffle index to the chunk
                        # print("Applying shuffle index to data chunk...")
                        # print(f"key: {key}, shuffle_indices: {shuffle_indices[:10]}, start: {start}, end: {end}")
                        shuffled_chunk = data_chunk[shuffle_indices]

                        # Write the shuffled chunk to the output dataset
                        # print(f"Writing shuffled chunk from {start} to {end}...")
                        out_dataset[start:end] = shuffled_chunk

                    print(f"Completed shuffling chunk from {start} to {end} in group '{group_name}'.")

    print(f"Shuffling complete. Shuffled data saved to {output_path}")

import h5py
import os
import time

def read_first_chunk(hdf5_path, group_name="train", chunk_size=400000):
    if not os.path.exists(hdf5_path):
        print(f"File {hdf5_path} does not exist.")
        return

    with h5py.File(hdf5_path, "r") as hdf_file:
        if group_name not in hdf_file:
            print(f"Group '{group_name}' does not exist in the file.")
            return
        
        group = hdf_file[group_name]
        first_dataset_name = list(group.keys())[0]  # Assuming you want to read the first dataset in the group
        
        print(f"Reading the first {chunk_size} entries from dataset '{first_dataset_name}' in group '{group_name}'...")
        
        start_time = time.time()
        
        # Reading the first chunk of data
        data_chunk = group[first_dataset_name][:chunk_size]
        
        end_time = time.time()
        print(f"Chunk loaded. Shape: {data_chunk.shape}")
        print(f"Time taken to load: {end_time - start_time:.4f} seconds")
        
        return data_chunk


if __name__ == "__main__":
    # Example usage
    previous_days = 7
    source_root = "./data"
    dataset_name = "Eastland County"
    # dataset_name = "El Paso County"
    output_path = f"./data/temp/{dataset_name}_transformer_{previous_days}_preprocessed.h5"
    #Check if the file exists
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping preprocessing.")
    else:
        preprocess_and_save_dataset(source_root, dataset_name, output_path, previous_days=previous_days, chunk_size=400000)

    # Shuffle the preprocessed data
    shuffle_input = input("Shuffle the preprocessed data? (y/n): ")
    if shuffle_input.lower() == "y":
        shuffle_output_path = f"./data/temp/{dataset_name}_transformer_{previous_days}_shuffled.h5"
        # read_first_chunk(output_path, group_name="train", chunk_size=4000)
        #if the file exists, delete it
        if os.path.exists(shuffle_output_path):
            os.remove(shuffle_output_path)
        # shuffle_hdf5_groups(output_path, shuffle_output_path, chunk_size=400000)
        shuffle_hdf5_groups_efficient(output_path, shuffle_output_path, chunk_size=400000)
    else:
        print("Shuffling skipped.")

