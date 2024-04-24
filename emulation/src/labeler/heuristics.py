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

import csv
import torch
import numpy as np


def outlier_detection(features):
    mean = torch.mean(features, axis=0)
    std = torch.std(features, axis=0)
    
    upper_bound = mean + 2 * std
    lower_bound = mean - 2 * std

    outlier_mask = (features > upper_bound) | (features < lower_bound)
    my_labels = torch.sum(outlier_mask.float(), dim=1).bool().int()

    return my_labels


def flooding_detection(features):
    sload = (features[:, 6] > 100000.0) # 6
    dload = (features[:, 7] == 0.0) # 7
    flooding_mask = sload & dload

    my_labels = flooding_mask.int()

    return my_labels


def allow_udp_con(df):
    my_labels = np.where((df['proto'] == 'udp') & (df['state'] == 'CON'), 0, -1)
    return my_labels


def allow_arp_int_or_con(df):
    my_labels = np.where((df['proto'] == 'arp') & \
                              ((df['state'] == 'INT') | (df['state'] == 'CON')), 0, -1)
    return my_labels


def allow_tcp_req_or_con(df):
    my_labels = np.where((df['proto'] == 'tcp') & \
                              ((df['state'] == 'REQ') | (df['state'] == 'CON')), 0, -1)
    return my_labels


def filter_proto(df):
    allowed_protos = ['arp', 'icmp', 'igmp', 'ospf', 'tcp', 'udp']
    my_labels = (~df['proto'].isin(allowed_protos)).map({True: 1, False: -1}).to_numpy()
    return my_labels


def filter_service(df):
    allowed_services = ['dns', 'ftp', 'ftp-data', 'ssh', 'smtp', 'http', '-']
    my_labels = (~df['service'].isin(allowed_services)).map({True: 1, False: -1}).to_numpy()
    return my_labels


def block_udp_flooding(df):
    my_labels = np.where((df['proto'] == 'udp') & \
                        #  (df['service'] == 'dns') & \
                              (df['state'] == 'INT') & \
                              (df['sload'] > 10000) & \
                              (df['dload'] == 0) & \
                              (df['sttl'] == 254), 1, -1)
    return my_labels


def block_malicious_http(df):
    my_labels = np.where((df['proto'] == 'tcp') & \
                              (df['service'] == 'http') & \
                            #   (df['sttl'] == 254) & \
                              (df['dttl'] == 252), 1, -1)
    return my_labels


def block_malicious_smtp(df):
    my_labels = np.where((df['proto'] == 'tcp') & \
                              (df['service'] == 'smtp') & \
                              (df['dttl'] == 252), 1, -1)
    return my_labels


def block_malicious_ftp(df):
    my_labels = np.where((df['proto'] == 'tcp') & \
                              (df['service'] == 'smtp') & \
                              (df['dttl'] == 252), 1, -1)
    return my_labels


def block_malicious_ftp_data(df):
    my_labels = np.where((df['proto'] == 'tcp') & \
                              (df['service'] == 'smtp') & \
                              (df['dttl'] == 252), 1, -1)
    return my_labels