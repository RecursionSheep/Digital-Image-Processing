import torch
import torch.nn as nn
import torch.nn.functional as F

class CRLR(nn.Module):
    def __init__(self, config):
        super(CRLR, self).__init__()
        self.w = nn.Parameter(torch.randn(config["batch_size"]) / 10)
        self.beta = nn.Parameter(torch.randn(config["input_size"] * config["bins"], config["output_size"]) / 10)
        self.params = [1e-5, 1e-3, 1e-5, 1e-2, 1e-7]
        self.steps = config["steps"]

    def forward(self, input, label_onehot=None, index=None):
        input = input.float()
        output = torch.softmax(torch.matmul(input, self.beta), dim=1)
        if label_onehot is None:
            return output
        output = torch.clamp(output, 1e-30, 1)
        W = self.w * self.w
        if index is None:
            loss = - (W * ((label_onehot * torch.log(output)).sum(1))).sum(0)
            # loss = - (label_onehot * torch.log(output)).sum()
            loss = loss + self.params[0] * (torch.norm(W) ** 2) + self.params[1] * (torch.norm(self.beta) ** 2) + self.params[2] * torch.norm(self.beta, p=1) + self.params[3] * ((W.sum() - 1) ** 2)
            # print(W.sum(), self.beta.sum(), torch.norm(W), torch.norm(self.beta))
            return output, loss
        loss = 0
        for i in range(index, index + self.steps):
            input_copy = input.clone()
            treatment = torch.zeros_like(input[:, i]).copy_(input[:, i])
            input_copy[:, i] = 0
            eps = 1e-5
            calc = lambda x, y: torch.matmul(x.T, W * y) / (torch.matmul(W.T, y) + eps)
            loss += self.params[4] * (torch.norm(calc(input_copy, treatment) - calc(input_copy, 1 - treatment)) ** 2)
        return output, loss