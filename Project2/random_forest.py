import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dataset import split_random, split_by_context

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

rf = RandomForestClassifier(n_estimators = 1000, random_state = 233)
rf.fit(train, train_label)
results = rf.predict(test)
acc = 0
for i in range(test_cnt):
	if results[i] == test_label[i]:
		acc += 1
acc = acc / test_cnt
print(acc)
