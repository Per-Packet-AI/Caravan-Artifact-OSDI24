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


caravan_gpu_cycle = [] 
caravan_accuracy_gain = []
cts_training_gpu_cycle = [] 
cts_training_accuracy_gain = []
oracle_gpu_cycle = []
oracle_accuracy_gain = []


with open(f'{caravan_home}/emulation/scripts/results/figure10a-experiment.jsonl', 'r') as result_file:
    for record in result_file:
        data = json.loads(record)
        if data["retraining_trigger"] == 1:
            caravan_gpu_cycle.append(float(data["my_gpu_total_time"]))
            caravan_accuracy_gain.append(float(data["my_improvement"]))
        else:
            cts_training_gpu_cycle.append(float(data["my_gpu_total_time"]))
            cts_training_accuracy_gain.append(float(data["my_improvement"]))
            oracle_gpu_cycle.append(float(data["gt_gpu_total_time"]))
            oracle_accuracy_gain.append(float(data["gt_improvement"]))

# plotting
colors = [
    '#004DAF',
    '#ED1B3A',
    '#FF9900',
    '#33A02C',
    '#FABEAF',
    '#AAD59B'
]

# Plotting the lines
# plt.figure()
fig, ax = plt.subplots()
ax.plot(caravan_gpu_cycle, caravan_accuracy_gain, marker='o', linewidth=3, markersize=10, label='Caravan', color=colors[0])
ax.plot(cts_training_gpu_cycle, cts_training_accuracy_gain, marker='o', linewidth=3, markersize=10, label='Continuous retraining with generated labels', color=colors[2])
ax.plot(oracle_gpu_cycle, oracle_accuracy_gain, marker='o', linewidth=3, markersize=10, label='Continuous retraining', color=colors[1])

# Adding title and labels
plt.xlabel('GPU compute time (seconds)')
plt.ylabel('Accuracy gain (F1 score)')
plt.xlim([0.0, 6.0])
plt.ylim([0.0, 0.5])


# bbox_props = dict(boxstyle="larrow", fc=(1,1,1), ec="grey", lw=2)
# t = plt.text(8.0, 0.15, "Better", ha="right", va="bottom", rotation=-45,
#             bbox=bbox_props, c='grey')
# bb = t.get_bbox_patch()
# bb.set_boxstyle("larrow", pad=0.05)

# Adding a legend
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=1)
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

# Save the plot
if not os.path.exists(f"{caravan_home}/emulation/scripts/figures"):
    os.makedirs(f"{caravan_home}/emulation/scripts/figures")
plt.savefig(f"{caravan_home}/emulation/scripts/figures/figure-10a.png")