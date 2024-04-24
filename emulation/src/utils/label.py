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
from numpy.random import default_rng


def examine_label_quality(labeled_flows_file_path):
    num_flows = 0
    num_malicious_flows = 0
    num_labeled = 0
    num_labeled_correct = 0

    with open(labeled_flows_file_path, "r") as labelf:
        flow_reader = csv.reader(labelf)
        next(flow_reader, None)  # skip the headers
        for flow in flow_reader:
            num_flows += 1
            if float(flow[-1]) == 1.0:
                num_malicious_flows += 1
            if float(flow[-2]) == 1.0:
                num_labeled += 1
                if float(flow[-1]) == 1.0:
                    num_labeled_correct += 1

    print("======================================================")
    print("Statistics for packet/flow labeling")
    print(f"total # flows                           {num_flows}")
    print(f"total # malicious flows                 {num_malicious_flows}")
    print(f"total # labeled flows                   {num_labeled}")
    print(f"total # correctly labeled flows         {num_labeled_correct}")


def mess_up_label(golden_label, percentage):
    num_reverse = int(golden_label.shape[0] * percentage)
    # reverse_indices = torch.randint(0, golden_label.shape[0], (num_reverse,))
    rng = default_rng()
    reverse_indices = torch.from_numpy(rng.choice(golden_label.shape[0], size=num_reverse, replace=False))
    mask = torch.zeros_like(golden_label)
    mask[reverse_indices] = 1
    messed_label = torch.logical_xor(golden_label, mask).type(torch.int64)
    return messed_label


def aggregate_labels_simple(labels_a, labels_b):
    aggregated_labels = np.full(labels_a.shape, -1)
    mask_a_minus1 = (labels_a == -1)
    mask_b_minus1 = (labels_b == -1)
    mask_both_01 = ((labels_a >= 0) & (labels_b >= 0))
    aggregated_labels[mask_a_minus1] = labels_b[mask_a_minus1]
    aggregated_labels[mask_b_minus1] = labels_a[mask_b_minus1]
    aggregated_labels[mask_both_01] = -1
    return aggregated_labels


# def mix_my_positive_samples_with_reserve_negative_samples():

#     # Store some negative samples for retraining later
#     negative_label_features_reserve = torch.zeros((0, streaming_data.features.shape[1])).cuda()

#     if online_learning_attacks == 0:
#         if negative_label_features_reserve is None:
#             negative_label_features_reserve = features
#         else:
#             negative_label_features_reserve = torch.cat((negative_label_features_reserve, features), 0)

#     # labeling (skip if we use golden labels)
#     my_labeler = FlowLabeler(heuristics = "flooding_detection")
#     my_labels = my_labeler.label(online_learning_unscaled_features)
#     positive_label_indices = torch.nonzero(my_labels)
#     positive_labels = my_labels[positive_label_indices].squeeze(1)
#     positive_label_features = online_learning_features[positive_label_indices].squeeze(1)
#     # choose some negative samples from our storage
#     negative_label_features = sample_data_by_number(negative_label_features_reserve, positive_label_features.shape[0])
#     if negative_label_features.shape[0] < positive_label_features.shape[0]:
#         my_dnn.eval()
#         previous_dnn.eval()
#         online_learning_unscaled_features, online_learning_features, online_learning_labels, online_learning_attacks = None, None, None, 0
#         logging.info("no enough negative samples, do not retrain")
#         continue
#     negative_labels = torch.zeros(negative_label_features.shape[0])
#     # import pdb; pdb.set_trace()
#     # my_labels = mess_up_label(online_learning_labels, 0.2)
#     my_features = torch.cat((positive_label_features, negative_label_features), 0)
#     my_labels = torch.cat((positive_labels, negative_labels), 0)