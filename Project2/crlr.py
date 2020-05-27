import numpy as np
import torch

course_train = np.load("course_train.npy")
train = course_train[:, 0 : 512]
train_context = course_train[:, 512]
train_label = course_train[:, 513]
size = train.shape[0]
train = train / np.max(train, axis = 0, keepdims = True)

binary = np.zeros((size, 512 * 4))
for i in range(size):
	for j in range(512):
		for k in range(1, 5):
			if train[i, j] <= (k / 4):
				binary[i, j * 4 + k - 1] = 1.
				break
train = binary

test = train.copy()
test_context = train_context.copy()
test_label = train_label.copy()
test_cnt = 0
for i in range(size):
	if (train_context[i] >= 7):
	#if (i % 10 >= 7):
		test[test_cnt, :] = train[i, :]
		test_context[test_cnt] = train_context[i]
		test_label[test_cnt] = train_label[i]
		test_cnt += 1
	else:
		train[i - test_cnt, :] = train[i, :]
		train_context[i - test_cnt] = train_context[i]
		train_label[i - test_cnt] = train_label[i]
train_cnt = size - test_cnt
train = train[0 : train_cnt, :]
test = test[0 : test_cnt, :]
train_context = train_context[0 : train_cnt]
train_label = train_label[0 : train_cnt]
test_context = test_context[0 : test_cnt]
test_label = test_label[0 : test_cnt]

x_in = torch.tensor(train, dtype = torch.float)
y_in = torch.tensor(train_label, dtype = torch.long)
y_in = y_in.reshape([train_cnt, -1])
one_hot = torch.zeros(train_cnt, 10).scatter_(1, y_in, 1)
beta = torch.randn(2048, 10, requires_grad = True)
w = torch.randn(train_cnt, requires_grad = True)
W = w * w
pred = torch.softmax(torch.matmul(x_in, beta), dim = 1)
loss = - (W * (one_hot * torch.log(pred)).sum(1)).sum(0)

lambda1 = 1e-2
lambda2 = 1e-3
lambda3 = 1e-3
lambda4 = 1e-3
lambda5 = 1e-1
lr = 0.01

for treatment in range(2048):
	x_minus = x_in.clone()
	x_minus[:, treatment] = 0.
	bal1 = torch.matmul(x_minus.T, W * x_in[:, treatment]) / torch.matmul(W.T, x_in[:, treatment])
	bal2 = torch.matmul(x_minus.T, W * (1 - x_in[:, treatment])) / torch.matmul(W.T, (1 - x_in[:, treatment]))
	loss = loss + lambda1 * (torch.norm(bal1 - bal2) ** 2)

loss = loss + lambda2 * (torch.norm(W) ** 2) + lambda3 * (torch.norm(beta) ** 2) + lambda4 * torch.norm(W, p = 1)
loss = loss + lambda5 * ((torch.sum(W) - 1) ** 2)

for it in range(100):
	print(loss)
	loss.backward()
	w.data = w.data - w.grad.data * lr
	beta.data = beta.data - beta.grad.data * lr
	w.grad.data.zero_()
	beta.grad.data.zero_()

x_val = torch.tensor(test, dtype = torch.float)
pred = torch.max(torch.softmax(torch.matmul(x_val, beta), dim = 1), dim = 1)
cnt = 0
for i in range(test_cnt):
	if pred[i] == test_label[i]:
		cnt += 1
print(cnt / test_cnt)

