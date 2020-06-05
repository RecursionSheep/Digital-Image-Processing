import torch
import torch.nn as nn
import torch.optim
import numpy as np
from torch.autograd import Variable
from mlp import MLP
from crlr_model import CRLR
from evaluation import single_label_accuracy, generate_metric

device = "cuda:0" if torch.cuda.is_available() else "cpu"
save_model_file = "./model.pkl"

def train(config):
    model = config["model"]
    optim = config["optim"]
    train_loader = config["train_loader"]
    valid_loader = config["valid_loader"]
    epoch_num = config["epoch_num"]
    model.to(device)
    max_valid_acc, max_epoch = 0, 0
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0.05, max_lr=0.5, cycle_momentum=False)
    for epoch in range(epoch_num):
        # print("\r{} / {}".format(epoch, epoch_num), end="")
        total_loss = 0
        acc_result, metrics = None, None
        for step, data in enumerate(train_loader):
            input, label = data
            input, label = Variable(input).to(device), Variable(label).to(device).long()
            # index = torch.randperm(input.size(0)).to(device)
            # mix_rate = np.random.beta(alpha, alpha)
            # mix_input = mix_rate * input + (1 - mix_rate) * input[index]
            # mix_label = (label, label[index])
            optim.zero_grad()
            output, loss = model(input, label)
            # output, loss = model(mix_input, mix_label, mix_rate)
            total_loss += loss.item()
            loss.backward()
            # for i in range(0, config["input_size"] * config["bins"], config["steps"]):
            #     output, loss = model(input, label, i)
            #     total_loss += loss.item()
            #     loss.backward()
            optim.step()
            # lr_scheduler.step()
            # acc_result = single_label_accuracy(output, label, acc_result)
            acc_result = single_label_accuracy(output, label.max(dim=1)[1], acc_result)
            metrics = generate_metric(acc_result)
            print("\rTrain Epoch {}: Loss = {} | {}".format(epoch, round(total_loss / (step + 1), 3), metrics), end="")
        total_loss = 0
        print()
        acc_result, metrics = None, None
        for step, data in enumerate(valid_loader):
            input, label = data
            input, label = Variable(input).to(device), Variable(label).to(device).long()
            output, loss = model(input, label)
            # output = model(input)
            total_loss += loss.item()
            # total_loss += 0
            # acc_result = single_label_accuracy(output, label, acc_result)
            acc_result = single_label_accuracy(output, label.max(dim=1)[1], acc_result)
            metrics = generate_metric(acc_result)
            print("\rValid Epoch {}: Loss = {} | {}".format(epoch, round(total_loss / (step + 1), 3), metrics), end="")
        if metrics["micro_precision"] > max_valid_acc:
            max_valid_acc = metrics["micro_precision"]
            max_epoch = epoch
            torch.save(model, save_model_file)
        print()
    # print()
    print("Best Accuracy: {} / Best Epoch: {}".format(max_valid_acc, max_epoch))

def init_model_optim(local_config):
    model = MLP(local_config)
    # model = CRLR(local_config)
    if "test_loader" in local_config:
        model = torch.load(save_model_file)
    optim = torch.optim.Adam(model.parameters(), lr=local_config["learning_rate"], weight_decay=local_config["weight_decay"])
    # optim = torch.optim.SGD(model.parameters(), lr=local_config["learning_rate"])
    return model, optim