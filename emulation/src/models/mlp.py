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

import torch
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def l1_regularization(model, lambda_l1):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return lambda_l1 * l1_norm


# the MLP model used in the Taurus paper
class AD_NSL_KDD(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        self.linear_relu_stack.apply(init_weights)

    def forward(self, x):
        sigmoid_score = self.linear_relu_stack(x)
        return sigmoid_score


class ID_CIC_IDS2017(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(80, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        self.linear_relu_stack.apply(init_weights)

    def forward(self, x):
        sigmoid_score = self.linear_relu_stack(x)
        return sigmoid_score


class ID_CIC_IDS2017_large(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(78, 128),
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
        self.linear_relu_stack.apply(init_weights)

    def forward(self, x):
        sigmoid_score = self.linear_relu_stack(x)
        return sigmoid_score

    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label
    

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
        self.linear_relu_stack.apply(init_weights)

    def forward(self, x):
        sigmoid_score = self.linear_relu_stack(x)
        return sigmoid_score

    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label


class ID_CIC_IDS2017_small(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear(80, 40),
            # nn.ReLU(),
            # nn.Linear(40, 10),
            # nn.ReLU(),
            # nn.Linear(10, 2),
            # nn.Softmax(dim=1)
            nn.Linear(78, 10),
            nn.ReLU(),
            nn.Linear(10, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        )
        self.linear_relu_stack.apply(init_weights)

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
        self.linear_relu_stack.apply(init_weights)

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


# the MLP model used in the data plane of the UNSW-NB15 dataset
# input features and architecture are consistent with the N3IC paper
class ID_UNSW_NB15_N3IC(nn.Module):
    def __init__(self, input_shape=20, neurons=[32, 16, 2]):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, neurons[0]),
            # nn.BatchNorm1d(num_features=neurons[0]),
            nn.ReLU(),
            nn.Linear(neurons[0], neurons[1]),
            # nn.BatchNorm1d(num_features=neurons[1]),
            nn.ReLU(),
            nn.Linear(neurons[1], neurons[2]),
            nn.Softmax(dim=1)
        )
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)
    
    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label


# the MLP model used in the data plane of the UNSW-IoT dataset
# input features and architecture are consistent with the N3IC paper
class UNSW_IoT_N3IC(nn.Module):
    def __init__(self, input_shape=16, neurons=[64, 32, 10]):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, neurons[0]),
            # nn.BatchNorm1d(num_features=neurons[0]),
            nn.ReLU(),
            nn.Linear(neurons[0], neurons[1]),
            # nn.BatchNorm1d(num_features=neurons[1]),
            nn.ReLU(),
            nn.Linear(neurons[1], neurons[2]),
            nn.Softmax(dim=1)
        )
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)
    
    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label


# the MLP model used as the labeler of the CIC-ISCX dataset
# input features and architecture are consistent with the DeepPacket paper
class CIC_ISCX_DeepPacket(nn.Module):
    def __init__(self, 
                 input_shape=23, 
                 num_classes=7,
                 neurons=[400, 300, 200, 100, 50]):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, neurons[0]),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(neurons[0], neurons[1]),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(neurons[1], neurons[2]),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(neurons[2], neurons[3]),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(neurons[3], neurons[4]),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(neurons[4], num_classes),
            nn.Softmax(dim=1)
        )
        # self.model = nn.Sequential(
        #     nn.Linear(input_shape, neurons[3]),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.05),
        #     nn.Linear(neurons[3], neurons[4]),
        #     nn.ReLU(),
        #     # nn.Dropout(p=0.05),
        #     nn.Linear(neurons[4], num_classes),
        #     nn.Softmax(dim=1)
        # )
        self.model.apply(init_weights)

    def forward(self, x):
        return self.model(x)
    
    def run_inference(self, input_features):
        score = self.forward(input_features)
        label = torch.argmax(score, dim=1)
        return score, label