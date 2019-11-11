import numpy as np
import random
from random import sample

#sgd with stumps
#iteration1
x_train1 = np.load('madelon/cv-train-X.0.npy')
y_train1 = np.load('madelon/cv-train-y.0.npy')
x_test1 = np.load('madelon/cv-heldout-X.0.npy')
y_test1 = np.load('madelon/cv-heldout-y.0.npy')
acc_train1, acc_heldOut1 = train_and_evaluate_sgd_with_stumps(x_train1, y_train1, x_test1, y_test1)

#iteration2
x_train2 = np.load('madelon/cv-train-X.1.npy')
y_train2 = np.load('madelon/cv-train-y.1.npy')
x_test2 = np.load('madelon/cv-heldout-X.1.npy')
y_test2 = np.load('madelon/cv-heldout-y.1.npy')
acc_train2, acc_heldOut2 = train_and_evaluate_sgd_with_stumps(x_train2, y_train2, x_test2, y_test2)

#iteration3
x_train3 = np.load('madelon/cv-train-X.2.npy')
y_train3 = np.load('madelon/cv-train-y.2.npy')
x_test3 = np.load('madelon/cv-heldout-X.2.npy')
y_test3 = np.load('madelon/cv-heldout-y.2.npy')
acc_train3, acc_heldOut3 = train_and_evaluate_sgd_with_stumps(x_train3, y_train3, x_test3, y_test3)

#iteration4
x_train4 = np.load('madelon/cv-train-X.3.npy')
y_train4 = np.load('madelon/cv-train-y.3.npy')
x_test4 = np.load('madelon/cv-heldout-X.3.npy')
y_test4 = np.load('madelon/cv-heldout-y.3.npy')
acc_train4, acc_heldOut4 = train_and_evaluate_sgd_with_stumps(x_train4, y_train4, x_test4, y_test4)

#iteration5
x_train5 = np.load('madelon/cv-train-X.4.npy')
y_train5 = np.load('madelon/cv-train-y.4.npy')
x_test5 = np.load('madelon/cv-heldout-X.4.npy')
y_test5 = np.load('madelon/cv-heldout-y.4.npy')
acc_train5, acc_heldOut5 = train_and_evaluate_sgd_with_stumps(x_train5, y_train5, x_test5, y_test5)

#Test
X_train = np.load('madelon/train-X.npy')
y_train = np.load('madelon/train-y.npy')
X_test = np.load('madelon/test-X.npy')
y_test = np.load('madelon/test-y.npy')
acc_train, stumps_test_acc = train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test)

stumps_train_acc = (acc_train1 + acc_train2 + acc_train3 + acc_train4 + acc_train5)/5
stumps_heldout_acc = (acc_heldOut1 + acc_heldOut2 + acc_heldOut3 + acc_heldOut4 + acc_heldOut5)/5

stumps_train_std = np.std([acc_train1, acc_train2, acc_train3, acc_train4, acc_train5])
stumps_heldout_std = np.std([acc_heldOut1, acc_heldOut2, acc_heldOut3, acc_heldOut4, acc_heldOut5])

stumps_confidence_train =  (2.776 * stumps_train_std/math.sqrt(5))
stumps_confidence_heldout =  (2.776 * stumps_heldout_std/math.sqrt(5))

print(stumps_confidence_train)
print(stumps_confidence_heldout)