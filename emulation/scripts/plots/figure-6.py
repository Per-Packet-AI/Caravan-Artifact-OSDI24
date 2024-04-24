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

my_accuracy_gain = []
my_token_count = []
cts_training_with_llm_gain = []
cts_training_with_llm_token_count = []

with open(f'{caravan_home}/emulation/scripts/results/figure6-experiment.jsonl', 'r') as result_file:
    for record in result_file:
        data = json.loads(record)
        
        if data["labeling_window_size"] == 1:
            cts_training_with_llm_gain.append(data["my_improvement"])
            cts_training_with_llm_token_count.append(data["tokens_cost"])
        else:
            my_accuracy_gain.append(data["my_improvement"])
            my_token_count.append(data["tokens_cost"])


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
plt.figure(figsize=(8,4))
plt.plot(my_token_count, my_accuracy_gain, marker='o', linewidth=3, markersize=10, label='Continuous retraining with rule cache', color=colors[0])
plt.plot(cts_training_with_llm_token_count, cts_training_with_llm_gain, marker='o', linewidth=3, markersize=10, label='Continuous retraining without rule cache', color=colors[1])

# Adding title and labels
plt.xlabel('# tokens used for labeling')
plt.ylabel('Accuracy gain (F1 score)')
# plt.xlim([0, 800000])
# plt.ylim([0.0, 0.5])

bbox_props = dict(boxstyle="larrow", fc=(1,1,1), ec="grey", lw=2)
t = plt.text(800000, 0.0, "Better", ha="right", va="bottom", rotation=-45,
            bbox=bbox_props, c='grey')
bb = t.get_bbox_patch()
bb.set_boxstyle("larrow", pad=0.05)
# ax.legend(loc='upper left', prop={'size':25}, frameon=True)
# ax.legend(loc='lower right', frameon=True)

# Adding a legend
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1)
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

# Save the plot
if not os.path.exists(f"{caravan_home}/emulation/scripts/figures"):
    os.makedirs(f"{caravan_home}/emulation/scripts/figures")
plt.savefig(f"{caravan_home}/emulation/scripts/figures/figure-6.png")