import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletCenterLoss(nn.Module):
    def __init__(self):
        super(TripletCenterLoss, self).__init__()
        # self.mse = nn.MSELoss()

    def forward(self, input, label, centroids):
        # input: (B, H), label: (B) centroids: (N, H)
        B, N = input.size(0), centroids.size(0)
        # 最小化输入到质心的距离 ||o_k - c_k||^2
        # inner_loss = self.mse(input, centroids[label])
        input_norm = torch.norm(input, dim=1) # (B)
        inner_loss = torch.ones_like(label).float() - torch.einsum("ij,ij->i", input, centroids[label]) / torch.einsum("i,i->i", input_norm, torch.norm(centroids[label], dim=1))
        # 最大化输入与其他质心的距离 ||o_k - (\sum_{j\ne i}c_j) / (N - 1)||^2
        centroids_sum = centroids.sum(dim=0).unsqueeze(0).repeat(B, 1)  # (B, H)
        outer_loss = torch.einsum("ij,ij->i", input, centroids_sum - centroids[label]) / torch.einsum("i,i->i", input_norm, torch.norm(centroids_sum - centroids[label], dim=1))
        return (inner_loss + outer_loss).mean()

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = [config["input_size"]] + config["hidden_size"]
        self.output_size = config["output_size"]
        assert len(self.hidden_size) == self.num_layers
        self.fc_stack = nn.Sequential()
        for i in range(self.num_layers - 1):
            self.fc_stack.add_module("fc_{}".format(i), nn.Linear(self.hidden_size[i], self.hidden_size[i+1]))
            self.fc_stack.add_module("relu_{}".format(i), nn.ReLU())
        self.attention = nn.Parameter(torch.ones(self.hidden_size[0]))
        self.bn = nn.BatchNorm1d(self.hidden_size[-1])
        self.dropout = nn.Dropout(0.5)
        self.fc_out = nn.Linear(self.hidden_size[-1], self.output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_metric = TripletCenterLoss()

    def forward(self, input, label=None):
        # input: (B, H), label: (B), output: (B, N)
        input = input.float()
        # input = torch.einsum("ij,j->ij", input, self.attention)
        middle = self.fc_stack(input.view(input.size(0), -1))
        # middle = self.bn(middle)
        # middle = self.dropout(middle)
        output = self.fc_out(middle)
        if label is None:
            return output
        loss = self.criterion(output, label)
        # loss = loss + (self.attention.sum() - self.hidden_size[0]) ** 2
        loss_ext = self.criterion_metric(middle, label, self.fc_out.weight.data)
        loss = loss + loss_ext
        return output, loss
    # def forward(self, input, label, rate=-1):
    #     middle = self.fc_stack(input.float().view(input.size(0), -1))
    #     output = self.fc_out(middle)
    #     loss = rate * self.criterion(output, label[0]) + (1 - rate) * self.criterion(output, label[1]) if rate != -1 else self.criterion(output, label)
    #     return output, loss