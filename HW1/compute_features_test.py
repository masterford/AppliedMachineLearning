train = [line.rstrip('\n') for line in open('badges/train.names.txt')]
test = [line.rstrip('\n') for line in open('badges/test.names.txt')]

x_train1 = compute_features(train)
y_train1 = np.load('badges/train.labels.npy')
x_test1 = compute_features(test)
y_test1 = np.load('badges/test.labels.npy')

#SGD
sgd_acc_train1, sgd_acc_featuretest = train_and_evaluate_sgd(x_train1, y_train1, x_test1, y_test1)

#DT
dt_acc_train1, dt_acc_featuretest = train_and_evaluate_decision_tree(x_train1, y_train1, x_test1, y_test1)

#Stumps
dt4_acc_train1, dt4_acc_featuretest = train_and_evaluate_decision_stump(x_train1, y_train1, x_test1, y_test1)

#SGD_Stumps
stumps_acc_train1, stumps_acc_featuretest = train_and_evaluate_sgd_with_stumps(x_train1, y_train1, x_test1, y_test1)

print(sgd_acc_train1, sgd_acc_featuretest)
print(dt_acc_train1,dt_acc_featuretest)
print(dt4_acc_train1,dt4_acc_featuretest)
print(stumps_acc_train1,stumps_acc_featuretest)