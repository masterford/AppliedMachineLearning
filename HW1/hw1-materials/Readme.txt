This directory contains the data and code for CIS 419/519 homework 1.

madelon:
  The Madelon dataset for the cross-validation experiments is contained within
  this directory. Each file is a serialized numpy array. train-X.npy and
  train-y.npy contain all of the training features and labels. Similarly,
  test-X.npy and test-y.npy contain the testing data (used for the last part
  of the experiment). The full training data has already been split into
  cross-validation splits for you. For example, cv-train-X.0.npy,
  cv-train-y.0.npy, cv-heldout-X.0.npy, and cv-heldout-y.0.npy have the data
  which you should use during the first cross-validation step.

  Each of the matrices can be loaded into memory with the np.load() function.
  For example,

      import numpy as np
      X_train = np.load('madelon/train-X.npy')
      y_train = np.load('madelon/train-y.npy')
      X_test = np.load('madelon/train-X.npy')
      y_test = np.load('madelon/train-y.npy')

  You can find the size of the matrices with the .shape data member
  (e.g. print(X_train.shape)). The full training features and labels should
  have size (2000, 500) and (2000,). The testing features and labels should
  have size (600, 500) and (600). The cross-validation training features and
  labels should have size (1600, 500) and (1600,). The cross-validation held-out
  features and labels should have size (400, 500) and (400,)


badges:
  This directory contains the Badge dataset. The txt files contain the names, one
  name per line. The npy files have the labels for the training and test names,
  which can be loaded the same way as the Madelon data. There should not be a
  npy file for the hidden-test.names.txt. There should be 1000 names and labels
  in both the training and test splits.


latex:
  This directory contains a Latex template that you can optionally use for the
  writeup.


hw1.ipynb
  This file is a template for your python code. There is one function, "compute_features",
  defined in this file which you are required to turn in via Gradescope.
