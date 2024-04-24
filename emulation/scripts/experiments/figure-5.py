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

    # this runs the experiment associated with figure 5
    subprocess.run(
        [
            "python3",
            f"{caravan_home}/emulation/benchmarks/continuous-retraining.py",

            "--application_type",
            "iot_traffic_classification",

            "--job_name",
            "figure5-experiment",

            "--labeler_type",
            "device_list",

            "--eval_frequency",
            "1000",

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
            f"{caravan_home}/emulation/datasets/unsw-iot-workload1.csv",

            # Output File
            "-o",
            f"{caravan_home}/emulation/scripts/results/figure5-experiment.jsonl",

            # Logging Directory
            "--logdir",
            f"{caravan_home}/emulation/logs/figure5",

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