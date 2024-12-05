from datasets import DatasetDict, Dataset, load_from_disk
import torch
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

# This is done on the GeoTron end. 