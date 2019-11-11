### Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import interp
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc






### Classification
def concat_and_shuffled(class0,class1, shuffled=True):
    rnd = np.random.RandomState(8)
    
    # Create Individual Label Vectors
    Y_0   = np.zeros(class0.shape[0])
    Y_1   = np.ones(class1.shape[0])
    
    # Concatenate to Complete Vectors
    Y     = np.concatenate([Y_0,Y_1])
    X     = np.concatenate([class0,class1])
    
    if shuffled:
        shuffled_indices = rnd.permutation(np.arange(Y.shape[0]))
        return X[shuffled_indices], Y[shuffled_indices]
    else:
        return X, Y




def SVM_gridSearch(X,Y, folds):
    C = [0.01, 0.1, 1, 10]
    K = ['linear', 'rbf']
    param_grid = {'C':C, 'kernel':K}
    grid_search = GridSearchCV(svm.SVC(gamma='scale',probability=True,class_weight='balanced'), param_grid, cv=folds, n_jobs=-1)
    grid_search.fit(X,Y)
    grid_search.best_params_
    return grid_search.best_params_




def RF_randomSearch(X,Y, folds):
    n_estimators      = [int(x) for x in np.linspace(start=70, stop=250, num=25)]
    max_features      = ['auto', 'sqrt']
    max_depth         = [int(x) for x in np.linspace(start=5, stop=100, num=10)]
    min_samples_leaf  = [1, 2, 5, 10]
    min_samples_split = [2, 5, 10, 20]
    bootstrap         = [False]
    max_depth.append(None)
    random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth,
                   'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}
    grid_search = RandomizedSearchCV(RandomForestClassifier(class_weight='balanced', random_state=0),
                                     random_grid, n_iter=100, cv=folds, random_state=0, n_jobs=-1)
    grid_search.fit(X,Y)
    grid_search.best_params_
    return grid_search.best_params_




def ROC(X_train,Y_train,X_test,Y_test,clf1,clf2=None):
    # Train/Val ROC for One/Two Classifiers
    clf1_TPRS_train = []
    clf1_AUCS_train = []
    clf1_mean_FPR_train = np.linspace(0, 0.1, 100)
    clf1_FPR_train, clf1_TPR_train, thresholds = roc_curve(Y_train, clf1.predict_proba(X_train)[:,1])
    clf1_TPRS_train.append(interp(clf1_mean_FPR_train, clf1_FPR_train, clf1_TPR_train))
    clf1_TPRS_train[-1][0] = 0.0
    clf1_ROC_AUC_train = auc(clf1_FPR_train, clf1_TPR_train)
    clf1_AUCS_train.append(clf1_ROC_AUC_train)
    
    clf1_TPRS_test = []
    clf1_AUCS_test = []
    clf1_mean_FPR_test = np.linspace(0, 0.1, 100)
    clf1_FPR_test, clf1_TPR_test, thresholds = roc_curve(Y_test, clf1.predict_proba(X_test)[:,1])
    clf1_TPRS_test.append(interp(clf1_mean_FPR_test, clf1_FPR_test, clf1_TPR_test))
    clf1_TPRS_test[-1][0] = 0.0
    clf1_ROC_AUC_test = auc(clf1_FPR_test, clf1_TPR_test)
    clf1_AUCS_test.append(clf1_ROC_AUC_test)
    
    if clf2:
        clf2_TPRS_train = []
        clf2_AUCS_train = []
        clf2_mean_FPR_train = np.linspace(0, 0.1, 100)
        clf2_FPR_train, clf2_TPR_train, thresholds = roc_curve(Y_train, clf2.predict_proba(X_train)[:,1])
        clf2_TPRS_train.append(interp(clf2_mean_FPR_train, clf2_FPR_train, clf2_TPR_train))
        clf2_TPRS_train[-1][0] = 0.0
        clf2_ROC_AUC_train = auc(clf2_FPR_train, clf2_TPR_train)
        clf2_AUCS_train.append(clf2_ROC_AUC_train)

        clf2_TPRS_test = []
        clf2_AUCS_test = []
        clf2_mean_FPR_test = np.linspace(0, 0.1, 100)
        clf2_FPR_test, clf2_TPR_test, thresholds = roc_curve(Y_test, clf2.predict_proba(X_test)[:,1])
        clf2_TPRS_test.append(interp(clf2_mean_FPR_test, clf2_FPR_test, clf2_TPR_test))
        clf2_TPRS_test[-1][0] = 0.0
        clf2_ROC_AUC_test = auc(clf2_FPR_test, clf2_TPR_test)
        clf2_AUCS_test.append(clf2_ROC_AUC_test)
    
    plt.figure(figsize=[6,5])
    plt.plot(clf1_FPR_train, clf1_TPR_train, lw=1.5, alpha=1.0, label="SVM AUC (Training) = %0.5f"%clf1_ROC_AUC_train, color="black")
    plt.plot(clf1_FPR_test, clf1_TPR_test, lw=1.5, alpha=1.0, label="SVM AUC (Validation) = %0.5f"%clf1_ROC_AUC_test,  color="crimson")
    if clf2:
        plt.plot(clf2_FPR_train, clf2_TPR_train, lw=1.5, alpha=1.0, label="RF AUC (Training) = %0.5f"%clf2_ROC_AUC_train, color="purple")
        plt.plot(clf2_FPR_test, clf2_TPR_test, lw=1.5, alpha=1.0, label="RF AUC (Validation) = %0.5f"%clf2_ROC_AUC_test,  color="orangered")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.2)

