import numpy as np
from sklearn.ensemble import RandomForestClassifier

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

rf = RandomForestClassifier(n_estimators = 1000, random_state = 233)
rf.fit(train, train_label)
results = rf.predict(test)
acc = 0
for i in range(test_cnt):
	if results[i] == test_label[i]:
		acc += 1
acc = acc / test_cnt
print(acc)
