from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

"""
    Trains a SGDClassifier on the training data and computes two accuracy scores, the
    accuracy of the classifier on the training data and the accuracy of the decision
    tree on the testing data.
        
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
def train_and_evaluate_sgd(X_train, y_train, X_test, y_test):
    model = SGDClassifier(loss='log', max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    y_heldPred = model.predict(X_test)
    acc_train = accuracy_score(y_train, y_pred)
    acc_heldOut = accuracy_score(y_test, y_heldPred)
    
    
    return acc_train, acc_heldOut