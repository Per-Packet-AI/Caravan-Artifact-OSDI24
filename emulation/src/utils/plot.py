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

import matplotlib.pyplot as plt
import os


def savefig(filename, fig):
    import time
    timestr = time.strftime("%Y%m%d-%H%M%S")
    fig.savefig(f'{filename}_time_{timestr}.pdf')


def plot_cdf(input_data, 
             figure_name, 
             figure_repo_path=f"{caravan_home}/figures/"):
    
    fig, ax = plt.subplots()
    input_data_sorted = sorted(input_data)
    x_axis_index = [(k + 1) / len(input_data_sorted) for k in range(len(input_data_sorted))]
    ax.plot(input_data_sorted, x_axis_index, color="red")
    plt.grid()
    savefig(os.path.join(figure_repo_path, figure_name), fig)