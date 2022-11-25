import read_rsdata
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from keras.utils import np_utils
import numpy as np
def svm(X,y,kernel):
    if kernel=='rbf':
        param_grid= {'kernel': ['rbf'], 'C': [1,10,100], 'gamma': [0.001,0.01,0.1,10]}
        base_estimator = SVC(kernel='rbf',C = 10, gamma = 0.01)
    else:
        param_grid= {'kernel': ['linear'], 'C': [1,10,100]}
        base_estimator = SVC(kernel='linear')
    #grid_search = GridSearchCV(estimator = base_estimator, cv=5, param_grid = param_grid)
    #grid_search.fit(X,y)
    #print(grid_search.best_params_)
    #print(pd.DataFrame(grid_search.cv_results_).head())
    scores = cross_val_score(base_estimator, X, y, cv=5) 
    return scores.mean(), scores.std()
    #return grid_search.best_params_, grid_search.best_score_, scores.mean(),scores.std()

def random_forest(X,y):
    rfc = RandomForestClassifier(n_estimators = 100, max_depth=30, random_state=42)
    #parameter_grid = {'max_depth' : [5, 8, 10, 15, 30, 60, 100],
    #               'n_estimators':[10, 20, 30]}

    #grid_search = GridSearchCV(estimator = rfc, cv=5, param_grid = parameter_grid)
    #grid_search.fit(X,y)
    #print(grid_search.best_params_)
    #print(pd.DataFrame(grid_search.cv_results_).head())
    scores = cross_val_score(rfc, X, y, cv=5) 
    return scores.mean(),scores.std()
    #return grid_search.best_params_,grid_search.best_score_, scores.mean(),scores.std()
    
   
def main():
    d1 = read_rsdata.prepare_input_data(None, data = 'hx', conv = '1D', split = 'kfold')
    d2 = read_rsdata.prepare_input_data(None, data = 'mx', conv = '1D', split = 'kfold')
    hx,cg,ct = d1() 
    mx,_,_ = d2()
    hx = hx.reshape(-1,30*1287)
    mx = mx.reshape(-1,5*4096)

    best_params_1, best_result_1,scores_acc_1,scores_std_1 =  svm(hx,cg,'rbf')
    best_params_2, best_result_2,scores_acc_2,scores_std_2 =  svm(hx,ct,'rbf')
    scores_acc_3,scores_std_3 =  svm(mx,cg,'rbf')
    scores_acc_4,scores_std_4 =  svm(mx,ct,'rbf')
    
    #best_result = [best_result_1,best_result_2,best_result_3,best_result_4]
    scores_acc = [scores_acc_3,scores_acc_4]
    scores_std = [scores_std_3,scores_std_4]
    #best_params = [best_params_1,best_params_2,best_params_3,best_params_4]

    print(scores_acc,scores_std)
    
#if __name__ == "__main__":
#    main()
    
    