# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 14:58:57 2018

@author: jake
"""

"""
# 
# Get Data and format
#
"""

import pylab as plt
from sklearn import svm
#from random import random, randint
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
#from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import os
import os.path
from flask import Flask, request, render_template, url_for, jsonify, redirect
from flask_jsglue import JSGlue
from tempfile import mkdtemp

app = Flask(__name__)
jsglue = JSGlue(app)

app.jinja_env.add_extension('jinja2.ext.loopcontrols')
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['TEMPLATES_AUTO_RELOAD'] = True

# get the path for the parent folder to save plot images 
#parent_folder = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))

# extract data from csv and obtain array of data sets
def run_main():
    data_set = [list(map(float, line.rstrip().split(","))) for line in open("input3.csv")]
    data_length = len(data_set)
    
    # A = X_1 B = X_2 Y_vals = y, colors = y separate into colors red vs blue for 1 vs 0 
    A = [data_set[index][0] for index in range(data_length)]
    B = [data_set[index][1] for index in range(data_length)]
    
    Y_vals = np.array([data_set[index][2] for index in range(data_length)])
    
    colors = np.array(['red' if Y_vals[i] == 1.0 else 'blue' for i in range(data_length)])
    
    # coors  = [[X_1, X_2], ..., [X_1^n, X_2^n]]
    coords = np.array([[A[i], B[i]] for i in range(data_length)])
    
    # full maps will be used to show the full prediction map for the classifiers
    # needs to the be split into X and Y for plt parameters
    full_map_X = []
    full_map_Y = []
    full_map_coordinates = []
    
    for x in range(0, 40):
        x_coord = x/10
    for y in range(0, 40):
        y_coord = y/10
        full_map_X.append(x_coord)
        full_map_Y.append(y_coord)
        full_map_coordinates.append([x_coord, y_coord])
        
    def show_map(A, B, colors, filename):
        plt.figure()
        plt.scatter(A, B, c = colors)
        #    plt.savefig(str(parent_folder) + filename, bbox_inches='tight')
        plt.show()
        plt.clf()
    
    def train_classifier(grid, X_train, y_train, X_test, y_test, filename):
        grid.fit(X_train, y_train)
        
        #best_score = grid.best_score_
        test_out = grid.predict(X_test)
        #full_map_test = grid.predict(full_map_coordinates)
        #show_map(full_map_X, full_map_Y, full_map_test, filename)
        
        #print(test_out)
        return { 'test_data': X_test, 'result': test_out }
        """
        num_errors = 0
        
        for index in range(len(y_test)):
            if y_test[index] != test_out[index]:
                num_errors += 1
        total_test = len(test_out)
        test_acc = (total_test - num_errors)/total_test
        
        return str(best_score) + ", " + str(test_acc)
        """
     
    def C_test():
    
        k_folds = 5
        C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]
        # get training and testing data, using SSS as index generator
        cv = StratifiedShuffleSplit(n_splits = k_folds, test_size = 0.40)
        #print(coords)
        for train_index, test_index in cv.split(coords, colors):
            X_train, X_test = coords[train_index], coords[test_index]
            y_train, y_test = colors[train_index], colors[test_index]
            
        #######################
        training_A = [X_train[index][0] for index in range(len(X_train))]
        training_B = [X_train[index][1] for index in range(len(X_train))]
        
        testing_A = [X_test[index][0] for index in range(len(X_test))]
        testing_B = [X_test[index][1] for index in range(len(X_test))]
        #show map for the cross validated training set
        #show_map(training_A, training_B, y_train, "/cross_validation.png")
        ##############################
        
        #X_train, X_test, y_train, y_test = \
        #train_test_split(coords, colors, test_size=.4, random_state=None)
        #print(X_test)
        def Lin_Kernel():
            
            C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]
            
            params = dict(C = C_vals)
            
            #def linear_test():
            
            sv = SVC(kernel = "linear")
            grid = GridSearchCV(sv, params)
            
            return train_classifier(grid, X_train, y_train, X_test, y_test, "/linear.png")
            #print(sv.score(X_test, y_test))
            #linear_test()
        
        def Poly_Kernel():
            
            C_vals = [0.1, 1, 3]
            degree = [4, 5, 6]
            gamma = [0.1, 0.5]
            
            params = dict(C = C_vals, degree = degree, gamma = gamma)
            
            grid = GridSearchCV(SVC(kernel = 'poly'), param_grid = params)
            
            return train_classifier(grid, X_train, y_train, X_test, y_test, "/polynomial.png")
        
        def RBF_Kernel():
            
            k_folds = 5
            
            C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]
            gamma = [0.1, 0.5, 1, 3, 6, 10]
            #cv = StratifiedShuffleSplit(n_splits = k_folds, test_size = 0.40, random_state = 0)
            
            params = dict(C = C_vals, gamma = gamma)
            
            grid = GridSearchCV(SVC(), param_grid = params)
            
            #print(grid.cv_results_)                
            rbf_result = train_classifier(grid, X_train, y_train, X_test, y_test, "/RBF.png")
            
            return rbf_result
            #print(grid.best_params_, grid.best_score_)
            #clf = svm.SVC(C = C_val)
            #(best_C, best_score) = (grid.best_params_['C'], grid.best_score_)
            #clf.fit(X_train, y_train)    
            
            #testing_A = [X_test[index][0] for index in range(len(X_test))]
            #testing_B = [X_test[index][1] for index in range(len(X_test))]
            
            # array of predicted colors based on the testing coords
        
            # find number of errors, against the known values of y_test vs the test_out prediction
        
            #return str(best_score) + ", " + str(test_acc)
        
            #plt.figure()
            #plt.scatter(testing_A, testing_B, c = test_out)
            #plt.show()
            #plt.clf()
            
            
        def Log_Regress():
            C = [0.1, 0.5, 1, 5, 10, 50, 100]
            
            params = dict(C = C)
            
            grid = GridSearchCV(LogisticRegression(solver = 'lbfgs'), param_grid = params)
            
            return train_classifier(grid, X_train, y_train, X_test, y_test, "/logistic.png")
        
        def KNN():
            n_neighbors = [x for x in range(1, 51)]
            leaf_size = [int(x*5) for x in range(1, 13)]
            #print(n_neighbors)
            params = dict(n_neighbors = n_neighbors, leaf_size = leaf_size)
            
            grid = GridSearchCV(KNeighborsClassifier(), param_grid = params)
            
            return train_classifier(grid, X_train, y_train, X_test, y_test, "/KNN.png")        
        
        def Dec_Tree():
            
            max_depth = [int(x) for x in range(1, 51)]
            min_samples_split = [int(x) for x in range(2, 11)]
            
            params = dict(max_depth = max_depth, min_samples_split = min_samples_split)
            
            grid = GridSearchCV(DecisionTreeClassifier(), param_grid = params)
            return train_classifier(grid, X_train, y_train, X_test, y_test, "/decision_tree.png")
        
        def Rndm_Forest():
        
            max_depth = [int(x) for x in range(1, 51)]
            min_samples_split = [int(x) for x in range(2, 11)]
        
            params = dict(max_depth = max_depth, min_samples_split = min_samples_split)
            
            grid = GridSearchCV(RandomForestClassifier(), param_grid = params)
            
            return train_classifier(grid, X_train, y_train, X_test, y_test, "/random_forest.png")
        
        final_result = RBF_Kernel()
        #print(final_result)
        return final_result
    """
    method1 = Lin_Kernel()
    #print(method1)
    method2 = Poly_Kernel()
    #print(method2)
    method3 = RBF_Kernel()
    #print(method3)
    method4 = Log_Regress()
    #print(method4)
    method5 = KNN()
    #print(method5)
    method6 = Dec_Tree()
    #print(method6)
    method7 = Rndm_Forest()
    """
    return C_test()
    #print(method7)
    #with open("output3.csv", "w+") as file:
    #    file.write("svm_linear, " + method1 +"\n")
    #    file.write("svm_polynomial, " + method2 + "\n")
    #    file.write("svm_rbf, " + method3 + "\n")
    #    file.write("logistic, " + method4 + "\n")
    #    file.write("knn, " + method5 + "\n")
    #    file.write("decision_tree, " + method6 + "\n")
    #    file.write("random_forest, " + method7 + "\n")
    #
    #file.close()

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html")

@app.route("/get_model/", methods=["POST", "GET"])
def get_model():
    data = request.args.get('myData', '')
    print(data)
    results = run_main()
    final_data = {"test_data" : results['test_data'].tolist(), 'result' : results['result'].tolist()}
    print(final_data)
    return jsonify(final_data)

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

@app.route('/shutdown', methods=['POST', 'GET'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

if __name__ == "__main__":
    app.run(debug=False)
#full data_set map
#show_map(A, B, colors, "/full_data_set.png")

#
#
#

