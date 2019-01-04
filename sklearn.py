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
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
import numpy as np
import os
import os.path
from flask import Flask, request, render_template, url_for, jsonify, redirect
from flask_jsglue import JSGlue
from tempfile import mkdtemp
import json

app = Flask(__name__)
jsglue = JSGlue(app)

app.jinja_env.add_extension('jinja2.ext.loopcontrols')
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['TEMPLATES_AUTO_RELOAD'] = True

full_map_X = []
full_map_Y = []
full_map_coordinates = []

sample_range = {'min_x': 0, 'max_x' : 4, 'min_y' : 0, 'max_y' : 4}
# get the path for the parent folder to save plot images 
#parent_folder = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir))

# extract data from csv and obtain array of data sets
def show_map(A, B, colors, filename):
    plt.figure()
    plt.scatter(A, B, c = colors)
    #    plt.savefig(str(parent_folder) + filename, bbox_inches='tight')
    plt.show()
    plt.clf()
    
def train_classifier(grid, X_train, y_train, full_map, best = True):
#    print(X_train, y_train)
    grid.fit(X_train, y_train)

    #print(coords, colors)
    if best:
        best_score = grid.best_score_
        best_params = grid.best_params_
#    sampleTest = [[1.1, 2.4], [4.8, 5.2], [2.7, 3.3], [7.1, 5.5]]
    test_out = grid.predict(full_map)
#    print(test_out)
    #print('map length : ', len(full_map))
    #print('test out length :', len(test_out))
    #full_map_test = grid.predict(full_map_coordinates)
    #show_map(full_map_X, full_map_Y, full_map_test, filename)
    
    #print(test_out)
    return { 'test_data': full_map, 'result': test_out, 'params' : best_params, 'score' : best_score }
    """
    num_errors = 0
    
    for index in range(len(y_test)):
        if y_test[index] != test_out[index]:
            num_errors += 1
    total_test = len(test_out)
    test_acc = (total_test - num_errors)/total_test
    
    return str(best_score) + ", " + str(test_acc)    
            """
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

"""
k_folds = 5
C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]
# get training and testing data, using SSS as index generator
cv = StratifiedShuffleSplit(n_splits = k_folds, test_size = 0.40)
"""
#print(coords)
"""
for train_index, test_index in cv.split(coords, colors):
    X_train, X_test = coords[train_index], coords[test_index]
    y_train, y_test = colors[train_index], colors[test_index]
    
#######################
training_A = [X_train[index][0] for index in range(len(X_train))]
training_B = [X_train[index][1] for index in range(len(X_train))]

testing_A = [X_test[index][0] for index in range(len(X_test))]
testing_B = [X_test[index][1] for index in range(len(X_test))]
"""
def set_map(range_vals):
    full_map_coordinates = []
    
    low_x = min(0, int(range_vals.get('min_x')))
    high_x = int(range_vals.get('max_x'))
    low_y = min(0, int(range_vals.get('min_y')))
    high_y = int(range_vals.get('max_x'))
    
    x_coords = np.linspace(low_x, high_x, 40).tolist()
    y_coords = np.linspace(low_y, high_y, 40).tolist()
    for x_coord in x_coords:
        for y_coord in y_coords:
            full_map_coordinates.append([x_coord, y_coord])
    return np.array(full_map_coordinates)
#    full_map = np.array(full_map_coordinates)
"""
for x in range(0, 40):
    x_coord = x/10
    for y in range(0, 40):
        y_coord = y/10
        full_map_X.append(x_coord)
        full_map_Y.append(y_coord)
        full_map_coordinates.append([x_coord, y_coord])
    
full_map = np.array(full_map_coordinates)    
"""
set_map(sample_range)

def Lin_Kernel(X_data, Y_data, full_map, low_cv = 1):
    
    C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]
    
    params = dict(C = C_vals)
    
    #def linear_test():
    
#    sv = SVC(kernel = "linear")
    sv = LinearSVC()   
    grid = GridSearchCV(sv, params, cv=low_cv)
    
    linear_result = train_classifier(grid, X_data, Y_data, full_map)
    
    return linear_result
    #print(sv.score(X_test, y_test))
    #linear_test()

def Poly_Kernel(X_data, Y_data, full_map, low_cv = 1):
    
    C_vals = [0.1, 1, 3]
    degree = [4, 5, 6]
    gamma = [0.1, 0.5]
    
    params = dict(C = C_vals, degree = degree, gamma = gamma)
      
    grid = GridSearchCV(SVC(kernel = 'poly'), param_grid = params, cv=2 )
#    grid = SVC(kernel='poly', gamma=2, C=1, degree=4)   
    poly_result = train_classifier(grid, X_data, Y_data, full_map)
    
    return poly_result

def RBF_Kernel(X_data, Y_data, full_map, low_cv = 1):   
#    k_folds = 5   
    C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]
    gamma = [0.1, 0.5, 1, 3, 6, 10]
    #cv = StratifiedShuffleSplit(n_splits = k_folds, test_size = 0.40, random_state = 0)
    
    params = dict(C = C_vals, gamma = gamma)
#    kernel = 1.0 * RBF(1.0)
#    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
    gpc = SVC()
    grid = GridSearchCV(gpc, param_grid = params, cv=low_cv )
    
    #print(grid.cv_results_)                
    rbf_result = train_classifier(grid, X_data, Y_data, full_map)
    
    return rbf_result
            
def Log_Regress(X_data, Y_data, full_map, low_cv = 1):
    C = [0.1, 0.5, 1, 5, 10, 50, 100]
    
    params = dict(C = C)
    
    grid = GridSearchCV(LogisticRegression(solver = 'lbfgs'), param_grid = params, cv=low_cv)
    
    log_result = train_classifier(grid, X_data, Y_data, full_map)
    
    return log_result

def KNN(X_data, Y_data, full_map, low_cv = 1, length = 51):
    max_n_neighbors = min(length, 51)
#    print('n_neighbors here....... ',max_n_neighbors)
    n_neighbors = [x for x in range(1, max_n_neighbors)]
    leaf_size = [int(x*5) for x in range(1, 13)]
    #print(n_neighbors)
    params = dict(n_neighbors = n_neighbors, leaf_size = leaf_size)
    
    grid = GridSearchCV(KNeighborsClassifier(), param_grid = params, cv=low_cv )
    
    KNN_result = train_classifier(grid, X_data, Y_data, full_map)
    return KNN_result

def Dec_Tree(X_data, Y_data, full_map, low_cv = 1, length = 51):
    max_n_neighbors = min(length, 51)
#    print('n_neighbors here....... ',max_n_neighbors)
    max_depth = [int(x) for x in range(1, max_n_neighbors)]
    min_samples_split = [int(x) for x in range(2, 11)]
    
    params = dict(max_depth = max_depth, min_samples_split = min_samples_split)
    
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid = params, cv=low_cv )
    decTree_result = train_classifier(grid, X_data, Y_data, full_map)
    return decTree_result

def Rndm_Forest(X_data, Y_data, full_map, low_cv = 1 ,length = 51):
    max_n_neighbors = min(length, 51)
#    print('n_neighbors here....... ',max_n_neighbors)
    max_depth = [int(x) for x in range(1, max_n_neighbors)]
    min_samples_split = [int(x) for x in range(2, 11)]

    params = dict(max_depth = max_depth, min_samples_split = min_samples_split)
       
    grid = GridSearchCV(RandomForestClassifier(n_estimators = 100), param_grid = params, cv=low_cv) 
    
    rndmForest_result = train_classifier(grid, X_data, Y_data, full_map)
    return rndmForest_result
    
def run_test(method, X_data, Y_data, low_cv = 1, length = 51, range_vals = 'Sample'):
    
    if method == 'Sample Data':
        return { 'test_data': coords, 'result': colors }
    
    
    if range_vals == 'Sample':
        full_map = set_map(sample_range)
    else:
#        print(range_vals)
        full_map = set_map( range_vals )
        
    final_result = None
    
    if method == 'Linear':
        final_result = Lin_Kernel(X_data, Y_data, full_map, low_cv )
    if method == 'Polynomial':
        final_result = Poly_Kernel(X_data, Y_data , full_map, low_cv )
    if method == 'RBF':
        final_result = RBF_Kernel(X_data, Y_data, full_map, low_cv )
    if method == 'Log Regression':
        final_result = Log_Regress(X_data, Y_data, full_map, low_cv )
    if method == 'KNN':
        final_result = KNN(X_data, Y_data, full_map, low_cv, length )
    if method == 'Tree':
        final_result = Dec_Tree(X_data, Y_data, full_map, low_cv , length )
    if method == 'Forest':
        final_result = Rndm_Forest(X_data, Y_data, full_map, low_cv, length )
        
    return final_result

def parse_request(data_Set):
    
    raw_data = json.loads(data_Set)
    min_x = float("inf")
    max_x = float("-inf")
    min_y = float("inf")
    max_y = float("-inf")
    
    for data in raw_data:
        min_x = min(float(min_x), float(data[0]))
        max_x = max(float(max_x), float(data[0]))
        min_y = min(float(min_y), float(data[1]))
        max_y = max(float(max_y), float(data[1]))
        
    x_y = [[float(raw_data[i][0]), float(raw_data[i][1])] for i in range( len(raw_data) )]
    
    labels = [raw_data[i][2] for i in range(len(raw_data))]
    
    unique_labels = set(labels)
    low_count = min([labels.count(label) for label in unique_labels])
    low_cv = min(5, low_count)
    
    range_vals = {'max_x' : max_x, 'min_x': min_x, 'max_y' : max_y, 'min_y' : min_y}
    return ( np.array(x_y), np.array(labels), low_cv, range_vals)

@app.route("/", methods=["POST", "GET"])
def index():
    return render_template("index.html")

@app.route("/get_model/", methods=["POST", "GET"])
def get_model():
    method = request.args.get('runMethod', '')
    data_set = request.args.get('data_set')
#    print("args ", data_set, " ", method)
    X_data = [[]]
    Y_data = []
    low_cv = 5
    length = 51
    vals = 'Sample'
    
    if data_set == 'Sample':
#        print("data set sample")
        X_data = coords
        Y_data = colors
    else:
        (X_data, Y_data, low_cv, vals) = parse_request(data_set)
        length = int(len(Y_data) / 2)
        
    results = run_test(method, X_data, Y_data, low_cv, length, vals)
    final_data = {"test_data" : results.get('test_data').tolist(), 
                  'result' : results.get('result').tolist(), 
                  'score' : results.get('score'),
                  'params' : results.get('params')}
    #print(final_data)
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

