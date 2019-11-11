import math
from random import sample
def train_and_evaluate_sgd_with_stumps(X_train, y_train, X_test, y_test):
    
    Ntrain = len(X_train) 
    k = len(X_train[0])  #50
    Neval = len(X_test)
    
    X_prime_train = np.zeros((Ntrain, 50))
    X_prime_eval = np.zeros((Neval, 50))
    for j in range(0, 50):
        model = DecisionTreeClassifier(criterion='entropy', max_depth=4)
        indices = range(0, k)
        sample = random.sample(indices, math.ceil(k/2))
        X_random = X_train[:,sample]
        model.fit(X_random, y_train)  #train with random features
        X_eval_random = X_test[:,sample]
        ptrain = model.predict(X_random)
        peval = model.predict(X_eval_random)
        X_prime_train[:,j] = ptrain.T
        X_prime_eval[:,j] = peval.T
    
    #Train SDG classifier
    sgd_model = SGDClassifier(loss='log', max_iter=10000)
    sgd_model.fit(X_prime_train, y_train)
    y_pred = sgd_model.predict(X_prime_train)
    y_heldPred = sgd_model.predict(X_prime_eval)
    acc_train = accuracy_score(y_train, y_pred)
    acc_heldOut = accuracy_score(y_test, y_heldPred)
    return acc_train, acc_heldOut