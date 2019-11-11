from sklearn.tree import DecisionTreeClassifier


def train_and_evaluate_decision_tree(X_train, y_train, X_test, y_test):
    """
    Trains an unbounded decision tree on the training data and computes two accuracy scores, the
    accuracy of the decision tree on the training data and the accuracy of the decision
    tree on the testing data.
    
    The decision tree should use the information gain criterion (set criterion='entropy')
    
    Parameters
    ----------
    X_train: np.array
        The training features of shape (N_train, k)
    y_train: np.array
        The training labels of shape (N_train)
    X_test: np.array
        The testing features of shape (N_test, k)
    y_test: np.array
        The testing labels of shape (N_test)
    
    Returns
    -------
    The training and testing accuracies represented as a tuple of size 2.
    """
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_heldPred = model.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred)
    acc_heldOut = accuracy_score(y_test, y_heldPred)
    return acc_train, acc_heldOut


def train_and_evaluate_decision_stump(X_train, y_train, X_test, y_test):
    """
    Trains a decision stump of maximum depth 4 on the training data and computes two accuracy scores, the
    accuracy of the decision stump on the training data and the accuracy of the decision
    tree on the testing data.
    
    The decision tree should use the information gain criterion (set criterion='entropy')
    
    Parameters
    ----------
    X_train: np.array
        The training features of shape (N_train, k)
    y_train: np.array
        The training labels of shape (N_train)
    X_test: np.array
        The testing features of shape (N_test, k)
    y_test: np.array
        The testing labels of shape (N_test)
    
    Returns
    -------
    The training and testing accuracies represented as a tuple of size 2.
    """
    model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_heldPred = model.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred)
    acc_heldOut = accuracy_score(y_test, y_heldPred)
    return acc_train, acc_heldOut