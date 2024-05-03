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
import argparse
import logging
import copy
import matplotlib.pyplot as plt
import json
import time

import torch
torch.manual_seed(42)
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import f1_score

from models.mlp import *
from dataset.dataset import *
from labeler.labeler import *
from utils.metrics import *
from utils.label import *
from utils.others import *


# user-defined retraining trigger
def retraining_trigger(window_index, retrain_flags, 
                       my_model_f1s_proxy, static_model_f1s_proxy,
                       retrain_frequency=None):
    # naive
    # return True
    # timing-based
    if (window_index - 1) % retrain_frequency == 0:
        return True
    else:
        return False
    
    # # accuracy-based
    # if len(retrain_flags) == 0:
    #     return True
    # # scenario 1: we have not retrained for last window, and there is drift
    # if retrain_flags[-1] == False:
    #     # if my_model_f1s_proxy[-1] - static_model_f1s_proxy[-1] < 0.2:
    #     if my_model_f1s_proxy[-1] - my_model_f1s_proxy[-2] < 0.1:
    #         return True
    #     else:
    #         return False
    # # scenario 2: we have retrained for last window, and we need/do not need more retrain
    # else: # retrain_flags[-1] == True:
    #     if abs(my_model_f1s_proxy[-1] -  my_model_f1s_proxy[-2]) < 0.1:
    #         return False
    #     else:
    #         return True


# determine if we reach the end of a window
def end_of_window(k, batch_size, window_size):
    return (k + 1) * batch_size % window_size == 0


# determine if we should retrain based on generated labels
def if_retrain(generated_labels):
    if generated_labels.shape[0] == 0 or \
       generated_labels.sum() == 0 or \
       generated_labels.sum() == generated_labels.shape[0]:
        return False
    else:
        return True


# NOTE: this is for binary classification only
# form a class-balanced online training set based on generated labels
def form_training_set(all_features, all_labels, generated_labels):
    
    positive_label_indices = generated_labels.nonzero().squeeze(1)
    negative_label_indices = (generated_labels == 0).nonzero().squeeze(1)
    
    if generated_labels.sum() == 0:
        return None, None
    # elif generated_labels.sum() < generated_labels.shape[0] * 0.5:
    elif positive_label_indices.shape[0] < negative_label_indices.shape[0]:
        num_negative_samples_to_keep = positive_label_indices.shape[0]
        negative_sample_to_keep_indices = negative_label_indices[torch.randperm(negative_label_indices.shape[0])[:num_negative_samples_to_keep]]
        negative_features_to_keep = all_features[negative_sample_to_keep_indices]
        negative_label_to_keep = generated_labels[negative_sample_to_keep_indices]
        return torch.cat((negative_features_to_keep, all_features[positive_label_indices]), 0), \
               torch.cat((negative_label_to_keep, generated_labels[positive_label_indices]), 0)
    else:
        num_positive_samples_to_keep = negative_label_indices.shape[0]
        positive_sample_to_keep_indices = positive_label_indices[torch.randperm(positive_label_indices.shape[0])[:num_positive_samples_to_keep]]
        positive_features_to_keep = all_features[positive_sample_to_keep_indices]
        positive_label_to_keep = generated_labels[positive_sample_to_keep_indices]
        return torch.cat((positive_features_to_keep, all_features[negative_label_indices]), 0), \
               torch.cat((positive_label_to_keep, generated_labels[negative_label_indices]), 0)


# initialize logging functions and logging files
def initialize_logging(path_to_log_file):
    # initialize logging
    file_handler = logging.FileHandler(filename=path_to_log_file)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(level=logging.DEBUG, 
                        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                        handlers=handlers)
    
    # disable logging of matplotlib-related modules
    logging.getLogger('matplotlib.pyplot').disabled = True
    logging.getLogger('matplotlib.font_manager').disabled = True


# make predictions with the labeler
def predict_with_labeler(labeler, data):
    my_labels = labeler.label(data)
    my_labels = torch.from_numpy(my_labels)
    # NOTE: for now, treat unlabeled samples as negative samples
    # random_mask = torch.randint(0, 2, (my_labels.shape[0],))
    # my_labels = my_labels * (my_labels >= 0) + random_mask * (my_labels < 0)
    my_labels[my_labels < 0] = 0
    return my_labels


# retrain a model based on given labeled dataset
def retrain_model(my_dnn, training_dataloader, total_epochs, optimizer, loss_fn, l1_reg_flag=False, l1_reg_param=0.001):
    my_dnn.train()
    for epoch_index in range(total_epochs):
        # process each mini-batch
        for _, (features, gt_label) in enumerate(training_dataloader):
            # clear the gradients
            optimizer.zero_grad()
            # run inference
            my_score, _ = my_dnn.run_inference(features)
            loss = loss_fn(my_score, gt_label)
            if l1_reg_flag:
                loss += l1_regularization(my_dnn, l1_reg_param)
            # backprop
            loss.backward()
            optimizer.step()
        
        if epoch_index == 0:
            logging.info(f"training loss at epoch 0 = {loss}")
    
    logging.info(f"training loss at epoch {epoch_index} = {loss}")

    return my_dnn, optimizer


# initialize the labeler based on user-provided info and knowledge sources
def initialize_labeler(args):

    if args.labeler_type == "LLM":
        labeler = LMLabeler(model_name= args.llm_model_name,
                            api_key = args.openai_api_key,
                            application_type = args.application_type)
        labeler.initialize_assistant()
    
    elif args.labeler_type == "DNN_classifier":
        labeler = DNNLabeler(args.labeler_dnn_class, # "ID_CIC_IDS2017_large_pforest", 
                             args.labeler_dnn_path, 
                             device = args.device)
    
    elif args.labeler_type == "device_list":
        labeler = DeviceListLabeler(device_list_path = args.device_list_path, 
                                    device = args.device)
    
    # deprecated, as of now
    elif args.labeler_type == "rules_or_heuristics":
        labeler = FlowLabeler(heuristics_function = [
                                                        "allow_udp_con", "allow_arp_int_or_con", \
                                                        "allow_tcp_req_or_con", "filter_proto", \
                                                        "filter_service", "block_udp_flooding", \
                                                        "block_malicious_http", "block_malicious_smtp", \
                                                        "block_malicious_ftp", "block_malicious_ftp_data"
                                                    ])
    
    else:
        print("Specified labeler type not supported")
        exit(0)

    return labeler


# main streaming function
def continuous_retrain(args):

    # initialize logging
    # if not os.path.exists("osdi24_ae_logs"):
    #     os.makedirs("osdi24_ae_logs")
    # initialize_logging(os.path.join("osdi24_ae_logs", args.job_name))

    # # initialize result output
    # if not os.path.exists("osdi24_ae_results"):
    #     os.makedirs("osdi24_ae_results")
    
    # load config information
    list_input_csv_path = args.list_flow_input_csv_file_path
    base_model_path = args.base_model_path
    device = args.device

    # load training and evaluation hyperparameters
    batch_size = args.batch_size
    eval_frequency = args.eval_frequency
    total_epochs = args.total_epochs
    learning_rate = args.learning_rate
    # set up L1 regularization if necessary
    l1_reg_flag = False
    l1_reg_param = 0.001

    # load feature names
    feature_names = args.feature_names
    
    # read input csv files (ML features and labels)
    streaming_data = None
    use_generated_labels = True if args.application_type == "iot_traffic_classification" else False
    for k, input_csv_path in enumerate(list_input_csv_path):
        current_csv_data = MyDataset(input_csv_path, 
                                     standardize=True, 
                                     normalize=False, 
                                     device=device, 
                                     if_header=0, 
                                     feature_names=feature_names,
                                     generated_labels=use_generated_labels)
        if k == 0:
            streaming_data = current_csv_data
        else:
            streaming_data.merge(current_csv_data)
    streaming_dataloader = DataLoader(streaming_data, batch_size=batch_size, shuffle=False)

    # load and initialize the base model
    # my_dnn = ID_UNSW_NB15_N3IC(input_shape=streaming_data.features.shape[1]).to(device)
    model_class = args.model_class
    model_input_shape = args.model_input_shape
    my_dnn_class = globals()[model_class]
    if model_input_shape:
        my_dnn = my_dnn_class(input_shape=model_input_shape).to(device)
    else:
        my_dnn = my_dnn_class().to(device)
    if base_model_path is not None:
        my_dnn.load_state_dict(torch.load(base_model_path))
    my_dnn.eval()
    
    # initialize the static model and the oracle model
    static_dnn = copy.deepcopy(my_dnn)
    static_dnn.eval()
    oracle_dnn = copy.deepcopy(my_dnn)
    oracle_dnn.eval()

    # initialize the optimizer
    # optimizer = torch.optim.SGD(my_dnn.parameters(), lr=learning_rate, momentum=0.9)
    # oracle_optimizer = torch.optim.SGD(oracle_dnn.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = torch.optim.Adam(my_dnn.parameters(), lr=learning_rate)
    oracle_optimizer = torch.optim.Adam(oracle_dnn.parameters(), lr=learning_rate)
    
    # initialize the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # NOTE: if necessary, adjust the loss function to take care of particular classes
    # loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 10.0]).to(device))

    # initialize the labeler
    labeler = initialize_labeler(args)

    # initialize variables for collecting accuracy stats for visualization
    my_model_f1s, my_model_recalls, my_model_pres, my_model_accs = [], [], [], []
    static_model_f1s, static_model_recalls, static_model_pres, static_model_accs = [], [], [], []
    oracle_model_f1s, oracle_model_recalls, oracle_model_pres, oracle_model_accs = [], [], [], []
    rules_f1s, rules_recalls, rules_pres, rules_accs = [], [], [], []
    labeler_f1s, labeler_recalls, labeler_pres, labeler_accs = [], [], [], []

    # initialize variables for collecting accuracy proxies stats
    my_model_f1s_proxy = []
    static_model_f1s_proxy = []

    # retrain flags: have we retrained for this window?
    retrain_flags = []
    use_retraining_trigger = True

    # initialize variables for collecting inference results for accuracy computation
    my_model_labels, static_model_labels, oracle_model_labels, gt_labels = \
        torch.zeros((0)).to(device), torch.zeros((0)).to(device), torch.zeros((0)).to(device), torch.zeros((0)).to(device)

    # initialize variables storing features and labels for online learning
    online_learning_features, online_learning_raw_features, online_learning_labels = \
        torch.zeros((0, streaming_data.features.shape[1])).to(device), \
        torch.zeros((0, streaming_data.features.shape[1])).to(device), \
        torch.zeros((0)).to(device)
    online_learning_generated_labels = torch.zeros((0)).to(device)

    # variables for profiling and measurement purposes
    my_gpu_total_time = 0.0
    gt_gpu_total_time = 0.0
    llm_total_tokens = 0
    llm_total_dollars = 0

    # variables related to rule cache
    enable_rule_cache = False
    use_rule_cache_every_x_windows = 7
    use_rule_cache_for_this_window = False
    rule_cache_function_def = None

    # start streaming
    for k, (features, gt_label, raw_features, generated_label) in enumerate(streaming_dataloader):

        # collect features and labels for online learning later
        online_learning_features = torch.cat((online_learning_features, features), 0)
        online_learning_raw_features = torch.cat((online_learning_raw_features, raw_features), 0)
        online_learning_labels = torch.cat((online_learning_labels, gt_label), 0)
        online_learning_generated_labels = torch.cat((online_learning_generated_labels, generated_label), 0)

        # ====================================================================
        # inference (on every input packet/flow)
        # run inference with updated model (trained with generated labels)
        _, my_label = my_dnn.run_inference(features)
        # run inference with static pre-trained model
        _, static_label = static_dnn.run_inference(features)
        # run inference with updated model (trained with dataset publisher labels)
        _, oracle_label = oracle_dnn.run_inference(features)
        # ====================================================================

        my_model_labels = torch.cat((my_model_labels, my_label), 0)
        static_model_labels = torch.cat((static_model_labels, static_label), 0)
        oracle_model_labels = torch.cat((oracle_model_labels, oracle_label), 0)
        gt_labels = torch.cat((gt_labels, gt_label), 0)
        
        # ====================================================================
        # compute and report accuracy at the end of a window
        if end_of_window(k, batch_size, eval_frequency):

            window_index = (k + 1) * batch_size / eval_frequency

            # if int(window_index) % 40 == 0:
            #     import pdb; pdb.set_trace()
            if enable_rule_cache:
                if (window_index - 1) % use_rule_cache_every_x_windows == 0:
                    use_rule_cache_for_this_window = False
                else:
                    use_rule_cache_for_this_window = True

            # compute f1 score, precision, recall, and accuracy
            # my_window_f1, my_window_pre, my_window_recall = compute_f1_full(my_model_labels, gt_labels)
            my_window_f1 = f1_score(gt_labels.cpu(), my_model_labels.cpu(), average='macro')
            my_window_accuracy = compute_acc(my_model_labels, gt_labels)
            # static_window_f1, static_window_pre, static_window_recall = compute_f1_full(static_model_labels, gt_labels)
            static_window_f1 = f1_score(gt_labels.cpu(), static_model_labels.cpu(), average='macro')
            static_window_accuracy = compute_acc(static_model_labels, gt_labels)
            # oracle_window_f1, oracle_window_pre, oracle_window_recall = compute_f1_full(oracle_model_labels, gt_labels)
            oracle_window_f1 = f1_score(gt_labels.cpu(), oracle_model_labels.cpu(), average='macro')
            oracle_window_accuracy = compute_acc(oracle_model_labels, gt_labels)

            # if window_index == 2.0:
            #     import pdb; pdb.set_trace()

            # if necessary, use the labeler or rules to do classification
            # rules_f1, _, _ = compute_f1_full(labeler.classify(online_learning_features, online_learning_generated_labels), gt_labels)
            # rules_accuracy = compute_acc(labeler.classify(online_learning_features, online_learning_generated_labels), gt_labels)
            # rules_f1s.append(rules_f1.cpu().item())
            # rules_accs.append(rules_accuracy.cpu().item())

            # update variables for collecting accuracy stats for visualization
            # my_model_f1s.append(my_window_f1.cpu().item())
            my_model_f1s.append(my_window_f1)
            # my_model_pres.append(my_window_pre.cpu().item())
            # my_model_recalls.append(my_window_recall.cpu().item())
            my_model_accs.append(my_window_accuracy.cpu().item())

            # static_model_f1s.append(static_window_f1.cpu().item())
            static_model_f1s.append(static_window_f1)
            # static_model_pres.append(static_window_pre.cpu().item())
            # static_model_recalls.append(static_window_recall.cpu().item())
            static_model_accs.append(static_window_accuracy.cpu().item())

            # oracle_model_f1s.append(oracle_window_f1.cpu().item())
            oracle_model_f1s.append(oracle_window_f1)
            # oracle_model_pres.append(oracle_window_pre.cpu().item())
            # oracle_model_recalls.append(oracle_window_recall.cpu().item())
            oracle_model_accs.append(oracle_window_accuracy.cpu().item())

            # report accuracy metrics
            logging.info(f"window {window_index}: k * batch_size = {k * batch_size}")
            logging.info(f"my f1 = {my_window_f1}\tmy acc = {my_window_accuracy}")
            logging.info(f"static f1 = {static_window_f1}\tstatic acc = {static_window_accuracy}")
            logging.info(f"oracle f1 = {oracle_window_f1}\toracle acc = {oracle_window_accuracy}")
        # ====================================================================

        # ====================================================================
            # labeling and retraining at the end of a window
            logging.info("labeling data from last window")
            
            # to use a DNN classifier as labeler
            if args.labeler_type == "DNN_classifier":
                valid_labels = labeler.label(online_learning_features)
                valid_features = online_learning_features

                # compute f1 proxies (using generated labels as ground truth)
                # my_window_f1_proxy, _, _ = compute_f1_full(my_model_labels, valid_labels)
                my_window_f1_proxy = f1_score(valid_labels.cpu(), my_model_labels.cpu(), average='macro')
                # static_window_f1_proxy, _, _ = compute_f1_full(static_model_labels, valid_labels)
                static_window_f1_proxy = f1_score(valid_labels.cpu(), static_model_labels.cpu(), average='macro')
                my_model_f1s_proxy.append(my_window_f1_proxy)
                static_model_f1s_proxy.append(static_window_f1_proxy)
            
            # logging.info(f"GT: # flows = {online_learning_labels.shape[0]}, # positive: {online_learning_labels.sum()}")
            # logging.info(f"large model: # flows = {valid_labels.shape[0]}, # positive: {valid_labels.sum()}")
            # logging.info(f"small model: # positive = {my_model_labels.sum()}")
            # logging.info(f"static model: # positive = {static_model_labels.sum()}")
            
            # to use a device list as labeler
            if args.labeler_type == "device_list":
                generated_labels = labeler.label(online_learning_features, online_learning_generated_labels)
                valid_label_indices = torch.nonzero(generated_labels != -1).squeeze(1)
                valid_labels = online_learning_labels[valid_label_indices]
                valid_features = online_learning_features[valid_label_indices]

                my_labels = my_model_labels[valid_label_indices]
                static_labels = static_model_labels[valid_label_indices]
                # compute f1 proxies (using generated labels as ground truth)
                # my_window_f1_proxy, _, _ = compute_f1_full(my_model_labels, valid_labels)
                my_window_f1_proxy = f1_score(valid_labels.cpu(), my_labels.cpu(), average='macro')
                # static_window_f1_proxy, _, _ = compute_f1_full(static_model_labels, valid_labels)
                static_window_f1_proxy = f1_score(valid_labels.cpu(), static_labels.cpu(), average='macro')
                my_model_f1s_proxy.append(my_window_f1_proxy)
                static_model_f1s_proxy.append(static_window_f1_proxy)

            # logging.info(f"GT: # flows = {online_learning_labels.shape[0]}, current class: {online_learning_labels[0].item()}")
            # logging.info(f"large model: # flows labeled = {valid_labels.shape[0]}")
            
            # to use an LLM as labeler
            if args.labeler_type == "LLM" and \
                ((not enable_rule_cache) or (enable_rule_cache and not use_rule_cache_for_this_window)):
                logging.info("using LLM for labeling this window")
                valid_labels = None
                
                # call LLM to obtain appropriate labels
                while valid_labels is None or valid_labels.shape[0] != gt_labels.shape[0]:
                    logging.info("labeling with LLMs...")
                    try:
                        valid_labels, api_response, tokens, price = labeler.label(online_learning_raw_features)
                        valid_labels = torch.tensor([float(k) for k in valid_labels]).cuda()
                    except:
                        logging.info("resend api request due to wrong output formatting")
                        continue
                
                valid_features = online_learning_features
                llm_total_tokens += tokens
                llm_total_dollars += price
        
                # generate rule function for the rule cache
                # now let's assume we generate rules each time we call the LLM
                if enable_rule_cache:
                    logging.info("generating rules for the rule cache")
                    compute_rules_function_input = torch.cat((online_learning_raw_features, \
                                                              valid_labels.unsqueeze(1)), 1)
                    rule_cache_function_def, rules_tokens, rules_price = labeler.compute_rules_function(compute_rules_function_input)
                    # try:
                    #     exec(rule_cache_function_def) # the rule cache func is called label_flow_with_rule_cache()
                    # except:
                    #     print("error when exec() on returned str def of rule cache func")
                    #     import pdb; pdb.set_trace()
                    llm_total_tokens += tokens
                    llm_total_dollars += price

            elif args.labeler_type == "LLM" and enable_rule_cache and use_rule_cache_for_this_window:
                logging.info("using rule cache for labeling this window")
                # try:
                #     exec(rule_cache_function_def) # the rule cache func is called label_flow_with_rule_cache()
                # except:
                #     print("error when exec() on returned str def of rule cache func")
                #     import pdb; pdb.set_trace()
                valid_labels = label_flows_with_rule_cache(online_learning_raw_features, rule_cache_function_def)

            # to skip labeling
            # else:
            #     valid_labels = gt_labels
            #     valid_features = online_learning_features     

            # if necessary, compute labeler f1 and accuracy
            # labeler_window_f1, labeler_window_pre, labeler_window_recall = compute_f1_full(valid_labels, gt_labels)
            if args.labeler_type != "device_list":
                labeler_window_f1 = f1_score(gt_labels.cpu(), valid_labels.cpu(), average='macro')
                labeler_window_accuracy = compute_acc(valid_labels, gt_labels)
                
                # store computed f1 scores and accuracies
                # labeler_f1s.append(labeler_window_f1.cpu().item())
                labeler_f1s.append(labeler_window_f1)
                # labeler_pres.append(labeler_window_pre.cpu().item())
                # labeler_recalls.append(labeler_window_recall.cpu().item())
                labeler_accs.append(labeler_window_accuracy.cpu().item())

                # logging.info(f"labeler f1 = {labeler_window_f1}\tlabeler pre = {labeler_window_pre}\tlabeler recall = {labeler_window_recall}\tlabeler acc = {labeler_window_accuracy}")
                logging.info(f"labeler f1 = {labeler_window_f1}\tlabeler acc = {labeler_window_accuracy}")
            # logging.info(f"GT: # flows = {online_learning_labels.shape[0]}, # positive: {online_learning_labels.sum()}")
            # logging.info(f"labeler: # flows = {valid_labels.shape[0]}, # positive: {valid_labels.sum()}")
            # logging.info(f"small model: # positive = {my_model_labels.sum()}")
            # logging.info(f"static model: # positive = {static_model_labels.sum()}")

            # Added on 11/28: before retraining, reset the model weights if f1 is too low
            # do not use it for iot traffic classification
            # if my_window_f1 < 0.2:
            #     my_dnn, optimizer = reset_model(my_dnn, optimizer, lr=learning_rate, static_model=static_dnn)
            # if oracle_window_f1 < 0.2:
            #     oracle_dnn, oracle_optimizer = reset_model(oracle_dnn, oracle_optimizer, lr=learning_rate, static_model=static_dnn)

            # do not retrain if labeled dataset is empty or too biased
            if not if_retrain(valid_labels):
                retrain_flags.append(False)
                logging.info("skip retraining due to empty or biased labeled dataset")
                # reset model status for inference
                online_learning_features, online_learning_raw_features, online_learning_labels = \
                    torch.zeros((0, streaming_data.features.shape[1])).to(device), \
                    torch.zeros((0, streaming_data.features.shape[1])).to(device), \
                    torch.zeros((0)).to(device)
                # reset variables for collecting inference results for accuracy computation
                my_model_labels, static_model_labels, oracle_model_labels, gt_labels = \
                    torch.zeros((0)).to(device), torch.zeros((0)).to(device), torch.zeros((0)).to(device), torch.zeros((0)).to(device)
                continue

            # retraining trigger: decide if we should retrain or not
            # accuracy-based retraining trigger
            if not use_retraining_trigger or \
                retraining_trigger(window_index, retrain_flags, my_model_f1s_proxy, static_model_f1s_proxy, args.retrain_frequency):

                # retrain the model (with generated labels from the labeler)
                if args.labeler_type != "device_list":
                    training_features, training_labels = form_training_set(online_learning_features, online_learning_labels, valid_labels)
                else:
                    training_features, training_labels = valid_features, valid_labels
                
                # if training_features is not None and valid_labels.sum() > eval_frequency * 0.05:
                if training_features is not None and training_features.nelement() > 0:
                    retrain_flags.append(True) # update retrain_flags
                    logging.info(f"# our retraining samples = {training_labels.shape[0]}")
                    training_dataset = OnlineDataset(training_features, training_labels, standardize=False, normalize=False, device=device)
                    training_dataloader = DataLoader(training_dataset, batch_size=512, shuffle=True)
                    logging.info("start training")

                    time_before_our_retrain = time.time()

                    my_dnn, optimizer = retrain_model(my_dnn,
                                                    training_dataloader,
                                                    total_epochs,
                                                    optimizer,
                                                    loss_fn,
                                                    l1_reg_flag=l1_reg_flag,
                                                    l1_reg_param=l1_reg_param)

                    time_after_our_retrain = time.time()
                    my_gpu_total_time += (time_after_our_retrain - time_before_our_retrain)
                else:
                    logging.info(f"# our retraining samples = 0 (skip retraining)")

            else:
                retrain_flags.append(False) # update retrain_flags
                logging.info(f"retraining trigger prevents us from retraining for this window")
                
            # # retrain the model (with generated labels from the labeler)
            # if args.labeler_type != "device_list":
            #     training_features, training_labels = form_training_set(online_learning_features, online_learning_labels, valid_labels)
            # else:
            #     training_features, training_labels = valid_features, valid_labels
            # # if training_features is not None and valid_labels.sum() > eval_frequency * 0.05:
            # if training_features is not None and training_features.nelement() > 0:
            #     logging.info(f"# our retraining samples = {training_labels.shape[0]}")
            #     training_dataset = OnlineDataset(training_features, training_labels, standardize=False, normalize=False, device=device)
            #     training_dataloader = DataLoader(training_dataset, batch_size=512, shuffle=True)
            #     logging.info("start training")

            #     time_before_our_retrain = time.time()

            #     my_dnn, optimizer = retrain_model(my_dnn,
            #                                     training_dataloader,
            #                                     total_epochs,
            #                                     optimizer,
            #                                     loss_fn,
            #                                     l1_reg_flag=l1_reg_flag,
            #                                     l1_reg_param=l1_reg_param)

            #     time_after_our_retrain = time.time()
            #     my_gpu_total_time += (time_after_our_retrain - time_before_our_retrain)
            # else:
            #     logging.info(f"# our retraining samples = 0 (skip retraining)")

            # retrain the model (with dataset publisher labels)
            if args.labeler_type != "device_list":
                oracle_training_features, oracle_training_labels = form_training_set(online_learning_features, online_learning_labels, online_learning_labels)
            else:
                oracle_training_features, oracle_training_labels = online_learning_features, online_learning_labels
            # if oracle_training_features is not None and online_learning_labels.sum() > eval_frequency * 0.05:
            if oracle_training_features is not None and oracle_training_features.nelement() > 0:
                logging.info(f"# oracle retraining samples = {oracle_training_labels.shape[0]}")
                oracle_training_dataset = OnlineDataset(oracle_training_features, oracle_training_labels, standardize=False, normalize=False, device=device)
                oracle_training_dataloader = DataLoader(oracle_training_dataset, batch_size=512, shuffle=True)
                
                time_before_gt_retrain = time.time()
                
                oracle_dnn, oracle_optimizer = retrain_model(oracle_dnn,
                                                            oracle_training_dataloader,
                                                            total_epochs,
                                                            oracle_optimizer,
                                                            loss_fn,
                                                            l1_reg_flag=l1_reg_flag,
                                                            l1_reg_param=l1_reg_param)

                time_after_gt_retrain = time.time()
                gt_gpu_total_time += (time_after_gt_retrain - time_before_gt_retrain)
            else:
                logging.info(f"# oracle retraining samples = 0 (skip retraining)")
            
            # reset model status for inference during the next window
            my_dnn.eval()
            oracle_dnn.eval()
            online_learning_features, online_learning_raw_features, online_learning_labels = \
                torch.zeros((0, streaming_data.features.shape[1])).to(device), \
                torch.zeros((0, streaming_data.features.shape[1])).to(device), \
                torch.zeros((0)).to(device)
            online_learning_generated_labels = torch.zeros((0)).to(device)

            # reset variables for collecting inference results for accuracy computation
            my_model_labels, static_model_labels, oracle_model_labels, gt_labels = \
                torch.zeros((0)).to(device), torch.zeros((0)).to(device), torch.zeros((0)).to(device), torch.zeros((0)).to(device)
        # ====================================================================

    # save key variables representing the results in a json file
    improvement_us = np.array(my_model_f1s) - np.array(static_model_f1s)
    improvement_us[improvement_us < 0.0] = 0.0
    improvement_gt = np.array(oracle_model_f1s) - np.array(static_model_f1s)
    improvement_gt[improvement_gt < 0.0] = 0.0
    improvement_us_avg = improvement_us.sum() / improvement_us[improvement_us != 0].size
    improvement_gt_avg = improvement_gt.sum() / improvement_gt[improvement_gt != 0].size
    print(f"avg gain for us: {improvement_us.sum() / improvement_us[improvement_us != 0].size}")
    print(f"avg gain for gt: {improvement_gt.sum() / improvement_gt[improvement_gt != 0].size}")
    if args.labeler_type == "LLM":
        print(f"token count for llm api: {llm_total_tokens}")
        print(f"dollar count for llm api: {llm_total_dollars}")
    print(f"gpu time for us: {my_gpu_total_time}")
    print(f"gpu time for gt: {gt_gpu_total_time}")
    
    result_json = {
        "retrain_frequency": args.retrain_frequency,
        "my_improvement": improvement_us_avg,
        "my_f1s": my_model_f1s,
        "static_f1s": static_model_f1s,
        "gt_improvement": improvement_gt_avg,
        "oracle_f1s": oracle_model_f1s,
        "labeler_f1s": labeler_f1s,
        "gpu_time": my_gpu_total_time,
    }
    
    # # Convert and write JSON object to file
    # with open(os.path.join("osdi24_ae_results", args.job_name + ".jsonl"), "a+") as outfile: 
    #     json.dump(result_json, outfile)
    #     outfile.write("\n")

    return result_json


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        help="Device type, e.g. cuda or cpu",
        default="cuda",
    )

    # application type and job name
    parser.add_argument(
        "--application_type",
        type=str,
        help="Type of application",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--job_name",
        type=str,
        help="Name of the job",
        required=True,
    )

    # everything related to the labeler
    parser.add_argument(
        "--labeler_type",
        type=str,
        help="Type of labeler",
        required=True,
    )
    parser.add_argument(
        "--llm_model_name",
        type=str,
        help="Type of LLM model (API) to be called",
        required=False,
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        help="OpenAI API key",
        required=False,
    )
    parser.add_argument(
        "--labeler_dnn_class",
        type=str,
        help="Class of the labeler DNN model",
        required=False,
    )
    parser.add_argument(
        "--labeler_dnn_path",
        type=str,
        help="Path to the labeler DNN model",
        required=False,
    )
    parser.add_argument(
        "--device_list_path",
        type=str,
        help="Path to the IoT device list",
        required=False,
    )

    # rule cache and retraining trigger related
    parser.add_argument(
        "--retrain_frequency",
        type=int,
        help="Retrain the model every X windows",
        default=10,
    )

    # retraining related
    parser.add_argument(
        "-e",
        "--total_epochs",
        type=int,
        help="Maximum number of epochs for training",
        default=10,
    )
    parser.add_argument(
        "-r",
        "--learning_rate",
        type=float,
        help="Initial learning rate",
        default=0.01,
    )

    # inference related
    parser.add_argument(
        "--eval_frequency",
        type=int,
        help="Evaluate F1 score/accuracy every X batches",
        default=70000,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size during inference",
        default=10,
    )

    # input data and base model
    parser.add_argument(
        "-i",
        "--list_flow_input_csv_file_path",
        nargs="*",
        type=str,
        help="List of path to the input flow csv file",
        required=True,
    )
    parser.add_argument(
        "--feature_names",
        nargs="*",
        type=str,
        help="Names of the input features",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        help="Name of model class",
        required=True,
    )
    parser.add_argument(
        "--model_input_shape",
        type=int,
        help="Number of input features",
    )
    parser.add_argument(
        "-m",
        "--base_model_path",
        type=str,
        help="Path to the base model path",
        required=False,
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output file name",
        required=True
    )

    parser.add_argument(
        "--logdir",
        type=str,
        help="Logging Directory",
        required=True
    )

    args = parser.parse_args()

    
    # initialize logging
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    initialize_logging(os.path.join(args.logdir, args.job_name))

    result_json = continuous_retrain(args,)
    

    # Convert and write JSON object to file
    if not os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))
    with open(args.output, "a+") as outfile: 
        json.dump(result_json, outfile)
        outfile.write("\n")