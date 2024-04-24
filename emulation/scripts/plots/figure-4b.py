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

labeling_window_sizes = []
generated_labels_improvements = []
publisher_labels_improvements = []

with open(f'{caravan_home}/emulation/scripts/results/figure4b-experiment.jsonl', 'r') as result_file:
    for record in result_file:
        data = json.loads(record)
        labeling_window_sizes.append(str(data["labeling_window_size"]))
        generated_labels_improvements.append(data["my_improvement"])
        publisher_labels_improvements.append(data["gt_improvement"])

# plotting
colors = [
    '#004DAF',
    '#ED1B3A',
    '#FF9900',
    '#33A02C',
    '#FABEAF',
    '#AAD59B'
]

bar_width = 0.3  # width of the bars
index = np.arange(len(labeling_window_sizes))  # the label locations
fig, ax = plt.subplots()

# Generate the bars for each category
rects1 = ax.bar(index - 0.5 * bar_width, generated_labels_improvements, bar_width, label='Retraining with LLM-generated labels', color=colors[0])
rects2 = ax.bar(index + 0.5 * bar_width, publisher_labels_improvements, bar_width, label='Retraining with ground-truth labels', color=colors[1])

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Labeling window size (# flows)')
ax.set_ylabel('Accuracy gain (F1)')
ax.set_xticks(index)
ax.set_xticklabels(labeling_window_sizes)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=1)
ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

# Adjust the layout
plt.tight_layout()

# Save the plot
if not os.path.exists(f"{caravan_home}/emulation/scripts/figures"):
    os.makedirs(f"{caravan_home}/emulation/scripts/figures")
plt.savefig(f"{caravan_home}/emulation/scripts/figures/figure-4b.png")