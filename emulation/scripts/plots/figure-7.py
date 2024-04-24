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

rule_cache_f1s = []

with open(f'{caravan_home}/emulation/scripts/results/figure7-experiment.jsonl', 'r') as result_file:
    for record in result_file:
        data = json.loads(record)
        rule_cache_f1s = data["labeler_f1s"][:20]
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

# locality of validation rule cache, and how they fail under data drift
retraining_windows = [int(k+1) for k in range(len(rule_cache_f1s))]

# Plotting the lines
plt.figure(figsize=(10,4))
plt.plot(retraining_windows, rule_cache_f1s, marker='o', linewidth=3, markersize=10, label='Label accuracy with labeling rule cache', color=colors[0])
plt.axvline(x = 1, linestyle="dashed", color = 'r', label = 'New rules in cache')
plt.axvline(x = 11, linestyle="dashed", color = 'black', label = 'Arrival of new data class')

# Adding title and labels
plt.xlabel('Continuous labeling window (over time)')
plt.ylabel('Accuracy (F1 score)')
plt.xticks(range(1, 21, 2))
plt.ylim([0.0, 1.05])

# bbox_props = dict(boxstyle="larrow", fc=(1,1,1), ec="grey", lw=2)
# t = plt.text(8.0, 0.0, "Better", ha="right", va="bottom", rotation=-45,
#             bbox=bbox_props, c='grey')
# bb = t.get_bbox_patch()
# bb.set_boxstyle("larrow", pad=0.05)

# Adding a legend
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1)
plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

# Save the plot
if not os.path.exists(f"{caravan_home}/emulation/scripts/figures"):
    os.makedirs(f"{caravan_home}/emulation/scripts/figures")
plt.savefig(f"{caravan_home}/emulation/scripts/figures/figure-7.png")