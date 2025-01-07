import os
import numpy as np
import pickle

with open('./mvp_phase_tool_phase_ant_tool_ant_57.pkl', 'rb') as f:
    all_info = pickle.load(f)


train_file_paths_30 = []
test_file_paths_27 = []

train_labels_30 = []
test_labels_27 = []

train_num_each_30 = []
test_num_each_27 = []

seq_num = len(all_info)
stat = np.zeros(12).astype(int)  # frame number of each phase
stat_train = np.zeros(12).astype(int)  # frame number of each phase
stat_test = np.zeros(12).astype(int)  # frame number of each phase

# for i in range(30,57):
for i in range(seq_num):
    if i<30:
        train_num_each_30.append(len(all_info[i]))
        for j in range(len(all_info[i])):
            train_file_paths_30.append(all_info[i][j][0])
            train_labels_30.append(all_info[i][j][1:])
            stat[all_info[i][j][1]] += 1
            stat_train[all_info[i][j][1]] += 1
    else:
        test_num_each_27.append(len(all_info[i]))
        for j in range(len(all_info[i])):
            test_file_paths_27.append(all_info[i][j][0])
            test_labels_27.append(all_info[i][j][1:])
            stat[all_info[i][j][1]] += 1
            stat_test[all_info[i][j][1]] += 1

print('Training videos numbers', len(train_num_each_30))
print(train_num_each_30)
print('testing videos numbers', len(test_num_each_27))
print(test_num_each_27)
print('video numbers: ', seq_num)

# print('frame number of each phase in MVP dataset: ', stat)
# print('total frames numbers in MVP dataset: ', stat.sum())
#
# print('frame number of each phase in MVP training set: ', stat_train)
# print('total frames numbers in MVP training set: ', stat_train.sum())
#
# print('frame number of each phase in MVP test set: ', stat_test)
# print('total frames numbers in MVP test set: ', stat_test.sum())

# print(np.max(np.array(test_labels_12)[:, 0]))
# print(np.min(np.array(test_labels_12)[:, 0]))


train_paths_labels_30 = []
test_paths_labels_27 = []

train_paths_labels_30.append(train_file_paths_30)
test_paths_labels_27.append(test_file_paths_27)

train_paths_labels_30.append(train_labels_30)
test_paths_labels_27.append(test_labels_27)

train_paths_labels_30.append(train_num_each_30)
test_paths_labels_27.append(test_num_each_27)


# train_val_test_paths_labels.append(test_file_paths_80)
# train_val_test_paths_labels.append(test_labels_80)
# train_val_test_paths_labels.append(test_num_each_80)

with open('train_paths_labels1_4task_30.pkl', 'wb') as f:
    pickle.dump(train_paths_labels_30, f)

with open('test_paths_labels1_4task_27.pkl', 'wb') as f:
    pickle.dump(test_paths_labels_27, f)

print('Done')