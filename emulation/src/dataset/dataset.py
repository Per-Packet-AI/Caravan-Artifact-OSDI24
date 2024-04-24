# *************************************************************************
#
# Copyright 2024 Qizheng Zhang (Stanford University),
#                Ali Imran (Purdue University),
#                Enkeleda Bardhi (Sapienza University of Rome),
#                Tushar Swamy (Unaffiliated),
#                Nathan Zhang (Stanford University),
#                Muhammad Shahbaz (Purdue University),
#                Kunle Olukotun (Stanford University)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# *************************************************************************

import copy
import sys
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.others import *


class MyDataset(Dataset):
    def __init__(self, root, 
                 standardize=True, 
                 normalize=False, 
                 device='cuda', 
                 if_header=None, 
                 feature_names=None,
                 generated_labels=None):
        
        if_header = None if if_header is None and feature_names is None else 0
        self.df = pd.read_csv(root, index_col=None, header=if_header)
        self.df.columns = [x.strip().lower() for x in self.df.columns]
        
        if generated_labels:
            self.generated_labels = torch.from_numpy(self.df["generated_label"].to_numpy()).type(torch.int64)

        if feature_names:
            cols_to_select = [x.strip().lower() for x in feature_names]
            cols_to_select.append("label")
            # ad-hoc fix for particular features; remove later
            if "total packet length" in feature_names and "number of packets" in feature_names:
                if "total packet length" not in self.df and "number of packets" not in self.df:
                    self.df.insert(4, "number of packets", self.df["total fwd packets"] + self.df["total backward packets"])
                    self.df.insert(7, "total packet length", self.df["total length of fwd packets"] + self.df["total length of bwd packets"])
            self.df = self.df[cols_to_select]

        self.data = self.df.to_numpy()
        self.features, self.labels = (torch.from_numpy(self.data[:,:-1]).type(torch.float32),
                                      torch.from_numpy(self.data[:,-1]).type(torch.int64))
        self.raw_features = copy.deepcopy(self.features)

        if not generated_labels:
            self.generated_labels = self.labels
        
        # data pre-processing
        ## standardize the data if needed
        if standardize:
            features_mean = torch.mean(self.features, dim=0).unsqueeze(0)
            features_std = torch.std(self.features, dim=0).unsqueeze(0)
            self.features = torch.nan_to_num((self.features - features_mean) / features_std)
        
        ## normalize the data if needed
        if normalize:
            scaler = MinMaxScaler()
            self.features = scaler.fit_transform(self.features)
            self.features = torch.from_numpy(self.features).type(torch.float32)
        
        # put on GPU if available
        self.features = self.features.to(device)
        self.raw_features = self.raw_features.to(device)
        self.labels = self.labels.to(device)
        self.generated_labels = self.generated_labels.to(device)
        
    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx], self.raw_features[idx, :], self.generated_labels[idx]
    
    def __len__(self):
        return len(self.data)

    # merge two input datasets (in-place)
    def merge(self, another_dataset):
        self.df = pd.concat([self.df, another_dataset.df])
        self.data = np.concatenate([self.data, another_dataset.data])
        self.features = torch.cat((self.features, another_dataset.features), 0)
        self.raw_features = torch.cat((self.raw_features, another_dataset.raw_features), 0)
        self.labels = torch.cat((self.labels, another_dataset.labels), 0)
        self.generated_labels = torch.cat((self.labels, another_dataset.generated_labels), 0)

    # merge two input datasets (not in-place)
    def merge_not_in_place(self, another_dataset):
        new_dataset = copy.deepcopy(self)
        new_dataset.df = pd.concat([self.df, another_dataset.df])
        new_dataset.data = np.concatenate([self.data, another_dataset.data])
        new_dataset.features = torch.cat((self.features, another_dataset.features), 0)
        new_dataset.raw_features = torch.cat((self.raw_features, another_dataset.raw_features), 0)
        new_dataset.labels = torch.cat((self.labels, another_dataset.labels), 0)
        new_dataset.generated_labels = torch.cat((self.labels, another_dataset.generated_labels), 0)
        return new_dataset


class OnlineDataset(Dataset):
    def __init__(self, features, labels, standardize=True, normalize=False, device='cuda'):
        self.features = features.type(torch.float32)
        self.labels = labels.type(torch.int64)
        # data pre-processing
        ## standardize the data
        if standardize:
            features_mean = torch.mean(self.features, dim=0).unsqueeze(0)
            features_std = torch.std(self.features, dim=0).unsqueeze(0)
            self.features = torch.nan_to_num((self.features - features_mean) / features_std)
        ## normalize the data if needed
        if normalize:
            scaler = MinMaxScaler()
            self.features = scaler.fit_transform(self.features)
            self.features = torch.from_numpy(self.features).type(torch.float32)
        # put on GPU if available
        self.features = self.features.to(device)
        self.labels = self.labels.to(device)

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx]
    
    def __len__(self):
        return self.features.shape[0]


# deprecated, as of now
class RawDataframe:
    def __init__(self, root):
        self.raw_df = pd.read_csv(root)


    def __len__(self):
        return self.raw_df.shape[0]
    

    # merge two input datasets (in-place)
    def merge(self, another_dataset):
        self.raw_df = pd.concat([self.raw_df, another_dataset.raw_df])


    def get_slice(self, start, end):
        return self.raw_df.iloc[start:end, :]