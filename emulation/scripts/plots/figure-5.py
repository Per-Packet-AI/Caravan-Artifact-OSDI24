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

import os
import json
import numpy as np
import matplotlib.pyplot as plt

caravan_home = os.getenv("CARAVAN_HOME")
if caravan_home is None:
    raise EnvironmentError("Could not find CARAVAN_HOME")

with open(f'{caravan_home}/emulation/scripts/results/figure5-experiment.jsonl', 'r') as result_file:
    for record in result_file:
        data = json.loads(record)
        my_model_f1s = data["my_f1s"]
        oracle_model_f1s = data["gt_f1s"]
        static_model_f1s = data["static_f1s"]
        labeler_f1s = data["labeler_f1s"]
        break

# plotting
colors = [
    '#004DAF',
    '#ED1B3A',
    '#FF9900',
    '#33A02C',
    '#FABEAF',
    '#AAD59B'
]

window_indices = [k+1 for k in range(len(my_model_f1s))]

plt.figure()
plt.plot(window_indices, my_model_f1s, marker='o', linewidth=3, markersize=10, label='Retraining with generated labels', color=colors[0])
plt.plot(window_indices, oracle_model_f1s, marker='o', linewidth=3, markersize=10, label='Retraining with ground-truth labels', color=colors[1])
plt.plot(window_indices, static_model_f1s, marker='o', linewidth=3, markersize=10, label='Offline model', color=colors[3])
plt.plot(window_indices, labeler_f1s, marker='o', linewidth=3, markersize=10, label='IoT device list as classifier', color=colors[2])
plt.axvline(x = 2, linestyle="dashed", color = 'black', label = 'Arrival of new data class')

# Adding title and labels
plt.xlabel('Continuous labeling window (over time)')
plt.ylabel('Accuracy (F1 score)')
plt.xlim([0, 7])
plt.ylim([0.0, 0.9])

# Adding a legend
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=2)
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

# Save the plot
if not os.path.exists(f"{caravan_home}/emulation/scripts/figures"):
    os.makedirs(f"{caravan_home}/emulation/scripts/figures")
plt.savefig(f"{caravan_home}/emulation/scripts/figures/figure-5.png")