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
import sys
import subprocess

caravan_home = os.getenv("CARAVAN_HOME")
if caravan_home is None:
    raise EnvironmentError("Could not find CARAVAN_HOME")


def run_experiment():

    labeling_window_sizes = [1000, 2000, 5000]
    use_retraining_trigger = 0

    # this runs the experiment associated with figure 10(c)
    for labeling_window_size in labeling_window_sizes:

        use_retraining_trigger = 1

        subprocess.run(
            [
                "python3",
                f"{caravan_home}/emulation/benchmarks/all.py",
                
                "--application_type",
                "iot_traffic_classification",
                
                "--job_name",
                "figure10c-experiment",

                "--labeler_type",
                "device_list",

                "--eval_frequency",
                f"{labeling_window_size}",

                "--retraining_trigger",
                f"{use_retraining_trigger}",

                "--batch_size",
                "1",
                "-r",
                "0.01",
                "--total_epochs",
                "30",
                "--model_class",
                "UNSW_IoT_N3IC",

                "-m",
                f"{caravan_home}/emulation/models/unsw-iot-in-network-dnn-5-classes.pt",

                "-i",
                f"{caravan_home}/emulation/datasets/unsw-iot-workload2.csv",

                # Output File
                "-o",
                f"{caravan_home}/emulation/scripts/results/figure10c-experiment.jsonl",

                # Logging Directory
                "--logdir",
                f"{caravan_home}/emulation/logs/figure10c",

                "--feature_names",
                "Dur",
                "SrcBytes",
                "DstBytes",
                "sTtl",
                "dTtl",
                "SrcLoad",
                "DstLoad",
                "SrcPkts",
                "DstPkts",
                "sMeanPktSz",
                "dMeanPktSz",
                "SIntPkt",
                "DIntPkt",
                "TcpRtt",
                "SynAck",
                "AckDat",
            ]
        )

        use_retraining_trigger = 0

        subprocess.run(
            [
                "python3",
                f"{caravan_home}/emulation/benchmarks/all.py",
                
                "--application_type",
                "iot_traffic_classification",
                
                "--job_name",
                "figure10c-experiment",

                "--labeler_type",
                "device_list",

                "--eval_frequency",
                f"{labeling_window_size}",

                "--retraining_trigger",
                f"{use_retraining_trigger}",

                "--batch_size",
                "1",
                "-r",
                "0.01",
                "--total_epochs",
                "30",
                "--model_class",
                "UNSW_IoT_N3IC",

                "-m",
                f"{caravan_home}/emulation/models/unsw-iot-in-network-dnn-5-classes.pt",

                "-i",
                f"{caravan_home}/emulation/datasets/unsw-iot-workload2.csv",

                # Output File
                "-o",
                f"{caravan_home}/emulation/scripts/results/figure10c-experiment.jsonl",

                # Logging Directory
                "--logdir",
                f"{caravan_home}/emulation/logs/figure10c",

                "--feature_names",
                "Dur",
                "SrcBytes",
                "DstBytes",
                "sTtl",
                "dTtl",
                "SrcLoad",
                "DstLoad",
                "SrcPkts",
                "DstPkts",
                "sMeanPktSz",
                "dMeanPktSz",
                "SIntPkt",
                "DIntPkt",
                "TcpRtt",
                "SynAck",
                "AckDat",
            ]
        )

    return


if __name__ == '__main__':

    run_experiment()
