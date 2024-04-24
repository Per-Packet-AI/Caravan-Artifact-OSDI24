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

from scapy.all import *
import numpy as np
from sklearn import metrics
import argparse

MAX_PKT_COUNT = 100
run_pkt_count = 0
pred_outputs = list()
true_labels = list()
f1_array  = list()


def to_float(x, e):
    c = abs(x)
    sign = 1 
    if x < 0:
        # convert back from two's complement
        c = x - 1 
        c = ~c
        sign = -1
    f = (1.0 * c) / (2 ** e)
    f = f * sign
    return f


# Parse received packets 
def parse_output(pkt,log_file):
    if Ether in pkt:
        data = pkt[Ether].payload
        # print()
        data = bytes(data)
        
        label  = data[32]
        output = data[33]
        valid_packet = data[34]


        global run_pkt_count, pred_outputs, true_labels

        if(valid_packet == 255):
            run_pkt_count += 1

            pred_outputs.append(output)
            true_labels.append(label)

            # Calculate metrics for every MAX_PKT_COUNT packets
            if run_pkt_count == MAX_PKT_COUNT:
                print("Received {0} packets.".format(MAX_PKT_COUNT))
                calc_metrics(pred_outputs, true_labels)
                run_pkt_count = 0

                with open(log_file, 'a') as file:
                    file.write(str(f1_array[-1]) + "\n")


# Computing DNN inference accuracy
def calc_metrics(pred_outputs=[], true_labels=[]):
    global f1_array
    accuracy = 100 * metrics.accuracy_score(true_labels, pred_outputs)
    precision = 100 * metrics.precision_score(true_labels, pred_outputs, average="weighted", labels=np.unique(pred_outputs))
    recall = 100 * metrics.recall_score(true_labels, pred_outputs, average="weighted")
    f1 = 100 * metrics.f1_score(true_labels, pred_outputs, average="weighted", labels=np.unique(pred_outputs))

    tn, fpo, fn, tp = metrics.confusion_matrix(true_labels, pred_outputs).ravel()

    print("Weighted Accuracy Across 2 Classes: {0:.2f}".format(accuracy))
    print("Weighted Precision Across 2 Classes: {0:.2f}".format(precision))
    print("Weighted Recall Across 2 Classes: {0:.2f}".format(recall))
    print("Weighted F1-Score Across 2 Classes: {0:.2f}".format(f1))

    f1_array.append(f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iface', type=str, required=True)
    parser.add_argument('--logfile', type=str, required=True)

    args = parser.parse_args()
    iface = args.iface
    log_file = args.logfile

    with open(log_file, 'w') as file:
        file.write("F1 Scores\n")
    
    # Sniff continuously on a selected iface
    while True:
        sniff(iface=iface, prn=lambda x:parse_output(x, log_file))


if __name__ == "__main__":
    
    main()
