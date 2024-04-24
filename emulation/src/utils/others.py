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

import torch
import pandas as pd
from numpy.random import default_rng


proto_conversion = {
    "ICMP": 1,
    "TCP": 6, 
    "UDP": 17,
    "unknown": 0
}


def sample_data_by_number(data, num_to_sample):
    if num_to_sample > data.shape[0]:
        return data
    rng = default_rng()
    sampled_indices = torch.from_numpy(rng.choice(data.shape[0], size=num_to_sample, replace=False))
    sampled_data = data[sampled_indices]
    return sampled_data


def process_unsw_nb15(path, drop_attack_cat=True):

    selected_columns = [
        'dur',
        'proto',
        'sbytes', 'dbytes',
        'sttl', 'dttl',
        'sload', 'dload',
        'spkts', 'dpkts',
        'smean', 'dmean',
        'sinpkt', 'dinpkt',
        'tcprtt', 'synack', 'ackdat',
        'ct_src_ltm', 'ct_dst_ltm',
        'ct_dst_src_ltm',
        'attack_cat',
        'label'
    ]

    my_df = pd.read_csv(path)
    try:
        my_df = my_df.drop(['id'], axis=1)
    except:
        pass
    cols = ['proto', 'service', 'state']
    for col in cols:
        my_df[col] =  my_df[col].astype('category')
        my_df[col] =  my_df[col].cat.codes
    
    my_df = my_df[selected_columns]
    # my_df["label"] = my_df['attack_cat'].apply(lambda x: 0.0 if x == "Normal" else 1.0)
    if drop_attack_cat:
        my_df = my_df.drop(columns="attack_cat")

    return my_df