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


def compute_accuracy_based_on_score(my_score, gt_label):
    my_label = torch.round(my_score)
    return (my_label == gt_label).int().sum() / gt_label.shape[0]


def compute_f1_based_on_score(my_score, gt_label, precision_recall=False, notify_all_false=True):
    my_label = torch.round(my_score)
    tp = ((my_label + gt_label) == 2.0).float().sum()
    fp = ((my_label - gt_label) == 1.0).float().sum()
    fn = (my_label != gt_label).float().sum() - fp
    precision = torch.nan_to_num(tp / (tp + fp))
    recall = torch.nan_to_num(tp / (tp + fn))
    f1 = torch.nan_to_num(2 * precision * recall / (precision + recall))
    if precision_recall:
        return f1, precision, recall
    return f1


def compute_acc(my_label, gt_label):
    return (my_label == gt_label).int().sum() / gt_label.shape[0]


def compute_f1(my_label, gt_label, precision_recall=False):
    tp = ((my_label + gt_label) == 2.0).float().sum()
    fp = ((my_label - gt_label) == 1.0).float().sum()
    fn = (my_label != gt_label).float().sum() - fp
    precision = torch.nan_to_num(tp / (tp + fp))
    recall = torch.nan_to_num(tp / (tp + fn))
    f1 = torch.nan_to_num(2 * precision * recall / (precision + recall))
    return f1


def compute_f1_full(my_label, gt_label, precision_recall=False):
    tp = ((my_label + gt_label) == 2.0).float().sum()
    fp = ((my_label - gt_label) == 1.0).float().sum()
    fn = (my_label != gt_label).float().sum() - fp
    precision = torch.nan_to_num(tp / (tp + fp))
    recall = torch.nan_to_num(tp / (tp + fn))
    f1 = torch.nan_to_num(2 * precision * recall / (precision + recall))
    return f1, precision, recall