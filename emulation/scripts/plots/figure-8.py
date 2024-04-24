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

with open(f'{caravan_home}/emulation/scripts/results/figure8-experiment.jsonl', 'r') as result_file:
    for record in result_file:
        data = json.loads(record)
        real_f1s = data["my_f1s"]
        proxy_f1s = data["my_f1s_proxy"]
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

retraining_windows = [k+1 for k in range(len(proxy_f1s))]

# Plotting the lines
plt.figure()
plt.plot(retraining_windows, proxy_f1s, linewidth=3, markersize=7, marker='o', label='Accuracy proxy (with generated labels)', color=colors[0])
plt.plot(retraining_windows, real_f1s, linewidth=3, markersize=7, marker='o', label='Real accuracy (with ground-truth labels)', color=colors[1])
for new_data_window_index in [k * 10 for k in range(1, 7)]:
    if new_data_window_index == 10:
        plt.axvline(x = new_data_window_index, linestyle="dashed", color = 'black', label = 'Arrival of new data class')
    else:
        plt.axvline(x = new_data_window_index, linestyle="dashed", color = 'black')

# Adding title and labels
plt.xlabel('Continuous labeling window (over time)')
plt.ylabel('Accuracy (F1 score)')
plt.xlim([0, 70])
plt.ylim([0.2, 1.05])

# Adding a legend
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=1)
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

# Save the plot
if not os.path.exists(f"{caravan_home}/emulation/scripts/figures"):
    os.makedirs(f"{caravan_home}/emulation/scripts/figures")
plt.savefig(f"{caravan_home}/emulation/scripts/figures/figure-8.png")