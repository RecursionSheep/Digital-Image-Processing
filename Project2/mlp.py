import numpy as np
import random
from sklearn.neural_network import MLPClassifier

course_train = np.load("course_train.npy")
train = course_train[:, 0 : 512]
train_context = course_train[:, 512]
train_label = course_train[:, 513]
size = train.shape[0]
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

'''
for i in range(train_cnt):
	if random.random() > .2:
		continue
	for j in range(i + 1, train_cnt):
		if random.random() > .2:
			continue
		if train_label[i] == train_label[j] and train_context[i] != train_context[j]:
			r = random.random()
			train = np.concatenate((train, np.array([train[i, :] * r + train[j, :] * (1 - r)])), axis = 0)
			train_label = np.append(train_label, np.array([train_label[i]]))
print(train.shape)
'''

lr = MLPClassifier(hidden_layer_sizes = (100), solver = 'sgd', alpha = 0.01, max_iter = 5000)
lr.fit(train, train_label)
results = lr.predict(test)
acc = 0
for i in range(test_cnt):
	if results[i] == test_label[i]:
		acc += 1
acc = acc / test_cnt
print(acc)
