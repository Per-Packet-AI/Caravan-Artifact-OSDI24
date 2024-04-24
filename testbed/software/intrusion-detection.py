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

from influxdb import InfluxDBClient
import time
import pypci
import struct
import numpy as np
import copy

# modules and libraries necessary for training
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


# Converts floating point number to a fixed point number
def c_to_fix(float_value):
    # Scaling by 2^8 (256) for 8 bits fractional part
    scaled_value = float_value * 65536

    # Converting to integer (rounding or truncating as necessary)
    integer_value = int(round(scaled_value))

    # Converting to 16-bit binary representation
    # Format the integer value as a binary string with 16 bits, padding with zeros if necessary
    binary_representation = format(integer_value, '032b')
    signed_integer = int(binary_representation,2)
    binary_data = struct.pack('>i', signed_integer)  # '>h' stands for big-endian 16-bit signed integer

    # Convert binary data to hexadecimal string
    hexadecimal_string = bin(int.from_bytes(binary_data, 'big'))[:]

    # Ensure the hexadecimal string is 16 bits long by zero-padding if necessary
    # hexadecimal_string = hexadecimal_string.zfill(16)

    # print(hex(int(hexadecimal_string,2)))
    return int(hexadecimal_string,2)


# the MLP model used as the labeler of the CIC-IDS2017/2018 dataset
# input features are consistent with the pForest paper
class ID_CIC_IDS2017_large_pforest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        sigmoid_score = self.linear_relu_stack(x)
        return sigmoid_score

    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label


# the MLP model used in the data plane of the CIC-IDS2017/2018 dataset
# input features are consistent with the pForest paper
class ID_CIC_IDS2017_small_pforest(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        sigmoid_score = self.linear_relu_stack(x)
        return sigmoid_score

    def result_after_first_layer(self, x):
        return self.linear_relu_stack[1](self.linear_relu_stack[0](x))

    def result_after_second_layer(self, x):
        y = self.result_after_first_layer(x)
        return self.linear_relu_stack[3](self.linear_relu_stack[2](y))

    def result_after_third_layer(self, x):
        z = self.result_after_second_layer(x)
        return self.linear_relu_stack[4](z)

    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label


# use this to generate the dataloader for online training
class OnlineDataset(Dataset):
    def __init__(self, features, labels, standardize=False, normalize=False, device='cpu'):
        self.features = features.type(torch.float32)
        self.labels = labels.type(torch.int64)

    def __getitem__(self, idx):
        return self.features[idx, :], self.labels[idx]
    
    def __len__(self):
        return self.features.shape[0]


# an example of a user-defined retraining trigger
def activate_retraining_trigger(retrain_flags, my_model_f1s_proxy):
    
    drift_detection_threshold = 0.2
    retrain_diminishing_gain_threshold = 0.1

    if len(retrain_flags) <= 1:
        return True
    
    # scenario 1: we have not retrained for last window, and there is drift
    if retrain_flags[-1] == False:
        if my_model_f1s_proxy[-2] - my_model_f1s_proxy[-1] > drift_detection_threshold:
            return True
        else:
            return False
    
    # scenario 2: we have retrained for last window, and we need/do not need more retrain
    else: # retrain_flags[-1] == True:
        if abs(my_model_f1s_proxy[-1] -  my_model_f1s_proxy[-2]) < retrain_diminishing_gain_threshold:
            return False
        else:
            return True


# retrain a model based on given labeled dataset
def retrain_model(my_dnn, training_dataloader, total_epochs, optimizer, loss_fn):
    
    my_dnn.train()
    
    for epoch_index in range(total_epochs):
        
        # process each mini-batch
        for _, (features, gt_label) in enumerate(training_dataloader):
            
            # clear the gradients
            optimizer.zero_grad()
            
            # run inference
            my_score, _ = my_dnn.run_inference(features)
            loss = loss_fn(my_score, gt_label)
            
            # backprop
            loss.backward()
            optimizer.step()
    
    return my_dnn, optimizer


# Setup access to the FPGA via PCIe
U250_VENDOR = 0x10ee
U250_GOLDEN_DEVICE = 0xd004
U250_DEVICE = 0x903f

SYS_CONFIG_BASE = 0x00000
SYS_CONFIG_HIGH = 0x01000
QDMA_SUBSYS_BASE = 0x01000
QDMA_SUBSYS_HIGH = 0x08000
CMAC_SUBSYS_BASE = 0x08000
CMAC_SUBSYS_HIGH = 0x10000
SPATIAL_SUBSYS_BASE = 0x10000
SPATIAL_SUBSYS_HIGH = 0x100000

u250_bar1 = None


def init(vendor, device):
    global u250_bar1
    # Open connection to the PCI device
    board = pypci.lspci(vendor=vendor, device=device)
    u250_board = board[0]
    # u250_board.vendor_id  # Read vendor id
    # u250_board.bar  # List bars
    u250_bar1 = u250_board.bar[1]
    return 0


def write(address, value):
    pypci.write(u250_bar1, address, struct.pack('<I', value))
    if(read(address) != value):
        print("Error: Could not write the value")
    return 0


def read(address):
    value = pypci.read(u250_bar1, address, 4)
    return int.from_bytes(value, "little")


# Set up connection to the influxdb database
INFLUXDB_HOST = "localhost"
INFLUXDB_PORT = 8086
INFLUXDB_DATABASE = "INFL_DB"

idbclient = InfluxDBClient(host=INFLUXDB_HOST, port=INFLUXDB_PORT, database=INFLUXDB_DATABASE)
prev_db_now = int(round(time.time() * 1000)) - 100

init(U250_VENDOR, U250_DEVICE)

# Enable CMACs
write(0x800C,1)
write(0x8014,1)

# write initial weights to the fpga
base_model_weights_path = "initial_model_weights.pt"
initial_weights = torch.load(base_model_weights_path, map_location=torch.device('cpu'))

# write initial weights to the fpga
layer_1_weights = initial_weights["linear_relu_stack.0.weight"].numpy()
layer_1_bias = initial_weights["linear_relu_stack.0.bias"].numpy()
layer_2_weights = initial_weights["linear_relu_stack.2.weight"].numpy()
layer_2_bias = initial_weights["linear_relu_stack.2.bias"].numpy()
layer_3_weights = initial_weights["linear_relu_stack.4.weight"].numpy()
layer_3_bias = initial_weights["linear_relu_stack.4.bias"].numpy()

# convert the weights into fixed point types here
all_params = np.concatenate((layer_1_weights.flatten(), layer_1_bias.flatten(), layer_2_weights.flatten(), layer_2_bias.flatten(), layer_3_weights.flatten(), layer_3_bias.flatten()))

enable_addr = int("100008",16)
base_addr = int("100010",16)

# disable register transfer flag
z = []

for y in all_params:
    z.append(c_to_fix(y))

write(enable_addr,0)
for i in range(len(z)):
    write((base_addr + i*4), z[i])
    # write((base_addr + i*4), 0)
# enable register transfer flag
write(enable_addr,1)

print("after writing initial weights")

# read initial weights from a file and initialize model
# base_model_weights_path = "initial_model_weights.pt"
my_dnn = ID_CIC_IDS2017_small_pforest()
my_dnn.load_state_dict(torch.load(base_model_weights_path, map_location=torch.device('cpu')))

# hyperparameters
labeling_window_size = 100
total_epochs = 30
learning_rate = 0.01
batch_size = 256

# define optimizer and loss function
optimizer = torch.optim.SGD(my_dnn.parameters(), lr=learning_rate, momentum=0.9)
loss_fn = torch.nn.CrossEntropyLoss()

print("after defining data-plane dnn, optimizer, and loss function")

# load labeling dnn here
labeler_dnn_weights_path = "labeler_dnn.pt"
labeler_dnn = ID_CIC_IDS2017_large_pforest()
labeler_dnn.load_state_dict(torch.load(labeler_dnn_weights_path, map_location=torch.device('cpu')))

print("after defining labeler dnn")

# retraining related
use_generated_labels = True
use_retraining_trigger = True
retrain_flags = []
my_model_f1s_proxy = []

# online training
data_buffer = []

while True:
    # add some delay here
    time.sleep(5)
    
    prev_db_now_temp = int(round(time.time() * 1000))
    prev_db_now_diff = prev_db_now_temp - prev_db_now
    prev_db_now = prev_db_now_temp

    # read data from the database
    results = idbclient.query(
                    'SELECT * FROM "INFL_DB"."autogen"."basic_tcp" WHERE time > now() - ' + str(prev_db_now_diff) + 'ms' )
    points = results.get_points()
    
    # points_duplicate = copy.deepcopy(points)
    # print("after points = results.get_points() ", len(list(points_duplicate)))
    # print("after points = results.get_points() ", len(list(points))," ", len(data_buffer))
    # add data into the buffer list
    # flows = []
    for p in points:
        network_flow = [
            p['input1'],
            p['input2'],
            p['input3'],
            p['input4'],
            p['input5'],
            p['input6'],
            p['input7'],
            p['input8'],
            p['input9'],
            p['input10'],
            p['input11'],
            p['input12'],
            p['input13'],
            p['input14'],
            p['input15'],
            p['input16'],
            p['output'],
            p['label']
        ]
        # if None in X_point:
        #     none_count += 1
        # else:
        #     flows.append([
        #         p['src_ip'],
        #         p['dst_ip'],
        #         p['src_port'],
        #         p['dst_port'],
        #         p['proto']
        #     ])
        #     X.append(X_point)
        if None not in network_flow:
            data_buffer.append(network_flow)
        # data_buffer.append(network_flow)
    
    print("after data_buffer.append(network_flow) ", len(data_buffer))

    # Do online training
    if len(data_buffer) > labeling_window_size:

        training_data = data_buffer
        training_data_tensor = torch.tensor(training_data)

        # retrieve our prediction label and ground truth label
        ground_truth_labels = training_data_tensor[:, -1]
        prediction_labels = training_data_tensor[:, -2]

        # label data in the current window with the labeler dnn      
        # _, generated_labels = labeler_dnn.run_inference(training_data_tensor)
        _, generated_labels = labeler_dnn.run_inference(training_data_tensor[:, :-2])

        if not use_generated_labels:
            generated_labels = ground_truth_labels

        # compute accuracy proxy for this window
        if use_retraining_trigger:
            f1_proxy = f1_score(generated_labels, prediction_labels)
            my_model_f1s_proxy.append(f1_proxy)

        if not use_retraining_trigger or \
            activate_retraining_trigger(retrain_flags, my_model_f1s_proxy):

            retrain_flags.append(True)
            print("activate retraining")
        
            # form a dataset for retraining
            # training_dataset = OnlineDataset(training_data_tensor[:, :-1], training_data_tensor[:, -1])
            training_dataset = OnlineDataset(training_data_tensor[:, :-2], generated_labels)
            training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

            # retrain
            my_dnn, optimizer = retrain_model(my_dnn, \
                                            training_dataloader, \
                                            total_epochs, \
                                            optimizer, \
                                            loss_fn)

            # obtain the new weights
            new_state_dict = my_dnn.state_dict()

            # reset data_buffer
            data_buffer = []

            # write the new weights back to the FPGA
            layer_1_weights = new_state_dict["linear_relu_stack.0.weight"].numpy()
            layer_1_bias = new_state_dict["linear_relu_stack.0.bias"].numpy()
            layer_2_weights = new_state_dict["linear_relu_stack.2.weight"].numpy()
            layer_2_bias = new_state_dict["linear_relu_stack.2.bias"].numpy()
            layer_3_weights = new_state_dict["linear_relu_stack.4.weight"].numpy()
            layer_3_bias = new_state_dict["linear_relu_stack.4.bias"].numpy()

            # convert the weights into fixed point types here
            all_params = np.concatenate((layer_1_weights.flatten(), layer_1_bias.flatten(), layer_2_weights.flatten(), layer_2_bias.flatten(), layer_3_weights.flatten(), layer_3_bias.flatten()))

            # Dummy Writes
            # write(0x800C,1)
            # write(0x8014,1)

            # enable_addr = int("100008",16)
            # base_addr = int("100010",16)

            start_time  = time.time()
            # disable register transfer flag

            z = []

            for y in all_params:
                z.append(c_to_fix(y))
            
            write(enable_addr,0)
            for i in range(len(z)):
                write((base_addr + i*4), z[i])
                # write((base_addr + i*4), 0)
            # enable register transfer flag
            write(enable_addr,1)
            print("Weights Update successful")
            end_time = time.time()

            print("Time Taken:", end_time - start_time)


        # skip retraining
        else:
            retrain_flags.append(False)
            print("skip retraining due to retraining trigger")