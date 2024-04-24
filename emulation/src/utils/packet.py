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

import numpy as np
from scapy.all import *

# Feature Header Protocol for Scapy
class FeatureHeader(Packet):

    name = "Feature Header"
    fields_desc=[StrField("flow_id", ""),  
                 StrField("timestamp", 0),  
                 LongField("duration", 0),  
                 ByteField("label", 0)]   # DNN Label (Ground Truth)

bind_layers(IP, FeatureHeader, proto=255)