import numpy as np
from sklearn.linear_model import LogisticRegression
#import torch
from dataset import split_random, split_by_context

def forward(x_in, one_hot, w, beta):
	W = w * w
	pred = torch.clamp(torch.softmax(torch.matmul(x_in, beta), dim = 1), 1e-5, 1)
	loss = - ((one_hot * torch.log(pred)).sum(1)).mean(0)
	'''lambda1 = 1e-2
	lambda2 = 1e-3
	lambda3 = 1e-3
	lambda4 = 1e-3
	lambda5 = 1e-1
	for treatment in range(2048):
		x_minus = x_in.clone()
		x_minus[:, treatment] = 0.
		bal1 = torch.matmul(x_minus.T, W * x_in[:, treatment]) / torch.matmul(W.T, x_in[:, treatment])
		bal2 = torch.matmul(x_minus.T, W * (1 - x_in[:, treatment])) / torch.matmul(W.T, (1 - x_in[:, treatment]))
		loss = loss + lambda1 * (torch.norm(bal1 - bal2) ** 2)

	loss = loss + lambda2 * (torch.norm(W) ** 2) + lambda3 * (torch.norm(beta) ** 2) + lambda4 * torch.norm(W, p = 1)
	loss = loss + lambda5 * ((torch.sum(W) - 1) ** 2)'''
	return loss

course_train = np.load("course_train.npy")
#train_data, test_data = split_random(course_train, None)
train_data, test_data = split_by_context(course_train, {"split_ratio": (5, 2)})
train = train_data[:, 0 : 512]
train_context = train_data[:, 512]
train_label = train_data[:, 513]
train_cnt = train.shape[0]
test = test_data[:, 0 : 512]
test_context = test_data[:, 512]
test_label = test_data[:, 513]
test_cnt = test.shape[0]

bins = 12

feature_max = np.maximum(np.max(train, axis = 0), np.max(test, axis = 0))
binary = np.zeros((train_cnt, 512 * bins))
for i in range(train_cnt):
	for j in range(512):
		for k in range(1, bins + 1):
			if train[i, j] <= (feature_max[j] * k / bins):
				binary[i, j * bins + k - 1] = 1.
				break
train = binary
binary = np.zeros((test_cnt, 512 * bins))
for i in range(test_cnt):
	for j in range(512):
		for k in range(1, bins + 1):
			if test[i, j] <= (feature_max[j] * k / bins):
				binary[i, j * bins + k - 1] = 1.
				break
test = binary

lr = LogisticRegression(C = 100)
lr.fit(train, train_label)
results = lr.predict(test)
acc = 0
for i in range(test_cnt):
	if results[i] == test_label[i]:
		acc += 1
acc = acc / test_cnt
print(acc)
exit()

x_in = torch.tensor(train, dtype = torch.float)
y_in = torch.tensor(train_label, dtype = torch.long)
y_in = y_in.reshape([train_cnt, -1])
one_hot = torch.zeros(train_cnt, 10).scatter_(1, y_in, 1)
beta = torch.randn(2048, 10, requires_grad = True)
w = torch.randn(train_cnt, requires_grad = True)

lr = 1.

for it in range(10000):
	loss = forward(x_in, one_hot, w, beta)
	print(loss.data)
	loss.backward()
	#w.data = w.data - w.grad.data * lr
	beta.data = beta.data - beta.grad.data * (lr / (it + 1))
	#w.grad.data.zero_()
	beta.grad.data.zero_()

x_val = torch.tensor(test, dtype = torch.float)
pred = torch.argmax(torch.softmax(torch.matmul(x_val, beta), dim = 1), dim = 1)
cnt = 0
for i in range(test_cnt):
	if pred.data[i] == test_label[i]:
		cnt += 1
print(cnt / test_cnt)

