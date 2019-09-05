from random import randint
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.gaussian_process import GaussianProcessClassifier
#from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import SVC
import os
import os.path
from flask import Flask, request, render_template, url_for, jsonify  #, redirect, g
from flask_jsglue import JSGlue
from tempfile import mkdtemp

from helpers import (BeautifulSoup, shuffle, threading, time, timedelta,
                     Parser, np, parse_request)
from svm import SVM_Helper

from config import getKeys
from threading import Thread, Lock
from pymongo import MongoClient
import datetime
from requests import get
import numpy as np

parser_dependencies = {"get": get}

svm_dependencies = {'np': np}

app = Flask(__name__)
jsglue = JSGlue(app)

app.jinja_env.add_extension('jinja2.ext.loopcontrols')
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['TEMPLATES_AUTO_RELOAD'] = True

svm_helper = SVM_Helper(**svm_dependencies)

full_map_X = []
full_map_Y = []
full_map_coordinates = []

sample_range = {'min_x': 0, 'max_x': 4, 'min_y': 0, 'max_y': 4}


def train_classifier(grid,
                     X_train,
                     y_train,
                     full_map,
                     best=True,
                     normalize=False):
    #    print(X_train, y_train)
    grid.fit(X_train, y_train)

    #print(coords, colors)
    if best:
        best_score = grid.best_score_
        best_params = grid.best_params_
#    sampleTest = [[1.1, 2.4], [4.8, 5.2], [2.7, 3.3], [7.1, 5.5]]
    test_out = None
    if normalize:
        test_out = grid.predict(preprocessing.scale(full_map))
    else:
        test_out = grid.predict(full_map)


#    print(test_out)
#print('map length : ', len(full_map))
#print('test out length :', len(test_out))
#full_map_test = grid.predict(full_map_coordinates)
#show_map(full_map_X, full_map_Y, full_map_test, filename)

#print(test_out)
    proba = grid.predict_proba(full_map)
    probability = [max(x.tolist()) for x in proba]
    return {
        'test_data': full_map,
        'result': test_out,
        'confidence': probability,
        'params': best_params,
        'score': best_score
    }

data_set = [
    list(map(float,
             line.rstrip().split(","))) for line in open("input3.csv")
]
data_length = len(data_set)

# A = X_1 B = X_2 Y_vals = y, colors = y separate into colors red vs blue for 1 vs 0
A = [data_set[index][0] for index in range(data_length)]
B = [data_set[index][1] for index in range(data_length)]

Y_vals = np.array([data_set[index][2] for index in range(data_length)])

colors = np.array(
    ['red' if Y_vals[i] == 1.0 else 'blue' for i in range(data_length)])

# coors  = [[X_1, X_2], ..., [X_1^n, X_2^n]]
coords = np.array([[A[i], B[i]] for i in range(data_length)])

# full maps will be used to show the full prediction map for the classifiers
# needs to the be split into X and Y for plt parameters


def set_map(range_vals):
    full_map_coordinates = []

    low_x = min(0, int(range_vals.get('min_x')))
    high_x = int(range_vals.get('max_x'))
    low_y = min(0, int(range_vals.get('min_y')))
    high_y = int(range_vals.get('max_x'))

    if range_vals.get('max_z', None) == None:
        x_coords = np.linspace(low_x, high_x, 35).tolist()
        y_coords = np.linspace(low_y, high_y, 35).tolist()
        for x_coord in x_coords:
            for y_coord in y_coords:
                full_map_coordinates.append([x_coord, y_coord])
        return np.array(full_map_coordinates)
    else:
        low_z = min(0, int(range_vals.get('min_z')))
        high_z = int(range_vals.get('max_z'))

        x_coords = np.linspace(low_x, high_x, 10).tolist()
        y_coords = np.linspace(low_y, high_y, 10).tolist()
        z_coords = np.linspace(low_z, high_z, 10).tolist()
        for x_coord in x_coords:
            for y_coord in y_coords:
                for z_coord in z_coords:
                    full_map_coordinates.append([x_coord, y_coord, z_coord])
        return np.array(full_map_coordinates)


#    full_map = np.array(full_map_coordinates)

set_map(sample_range)


def Lin_Kernel(training_data, label_data, full_map, low_cv=1):

    C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]

    params = dict(C=C_vals)

    #def linear_test():

    #    sv = SVC(kernel = "linear")
    sv = SVC(kernel='linear', probability=True)
    grid = GridSearchCV(sv, params, cv=low_cv)

    linear_result = train_classifier(grid, training_data, label_data, full_map)

    return linear_result
    #print(sv.score(X_test, y_test))
    #linear_test()


def Poly_Kernel(training_data, label_data, full_map, low_cv=1):

    C_vals = [0.1, 1, 3]
    degree = [4, 5, 6]
    gamma = [0.1, 0.5]

    params = dict(C=C_vals, degree=degree, gamma=gamma)

    grid = GridSearchCV(SVC(kernel='poly', probability=True),
                        param_grid=params,
                        cv=2)
    #    grid = SVC(kernel='poly', gamma=2, C=1, degree=4)
    poly_result = train_classifier(grid, training_data, label_data, full_map)

    return poly_result


def RBF_Kernel(training_data, label_data, full_map, low_cv=1):
    #    k_folds = 5
    C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]
    gamma = [0.1, 0.5, 1, 3, 6, 10]
    #cv = StratifiedShuffleSplit(n_splits = k_folds, test_size = 0.40, random_state = 0)

    params = dict(C=C_vals, gamma=gamma)
    #    kernel = 1.0 * RBF(1.0)
    #    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
    gpc = SVC(probability=True)
    grid = GridSearchCV(gpc, param_grid=params, cv=low_cv)

    #print(grid.cv_results_)
    rbf_result = train_classifier(grid, training_data, label_data, full_map)

    return rbf_result


def Log_Regress(training_data, label_data, full_map, low_cv=1):
    C = [0.1, 0.5, 1, 5, 10, 50, 100]

    params = dict(C=C)
    # any method that uses a gradient descent should normalize data set
    training_data = preprocessing.scale(training_data)
    grid = GridSearchCV(LogisticRegression(solver='lbfgs'),
                        param_grid=params,
                        cv=low_cv)
    log_result = train_classifier(grid, training_data, label_data, full_map,
                                  True, True)

    return log_result


def KNN(training_data, label_data, full_map, low_cv=1, length=51):
    max_n_neighbors = min(length, 51)
    #    print('n_neighbors here....... ',max_n_neighbors)
    n_neighbors = [x for x in range(1, max_n_neighbors)]
    leaf_size = [int(x * 5) for x in range(1, 13)]
    #print(n_neighbors)
    params = dict(n_neighbors=n_neighbors, leaf_size=leaf_size)
    #probability=True
    grid = GridSearchCV(KNeighborsClassifier(), param_grid=params, cv=low_cv)

    KNN_result = train_classifier(grid, training_data, label_data, full_map)
    return KNN_result


def Dec_Tree(training_data, label_data, full_map, low_cv=1, length=51):
    max_n_neighbors = min(length, 51)
    #    print('n_neighbors here....... ',max_n_neighbors)
    max_depth = [int(x) for x in range(1, max_n_neighbors)]
    min_samples_split = [int(x) for x in range(2, 11)]

    params = dict(max_depth=max_depth, min_samples_split=min_samples_split)

    grid = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=low_cv)

    decTree_result = train_classifier(grid, training_data, label_data,
                                      full_map)
    return decTree_result


def Rndm_Forest(training_data, label_data, full_map, low_cv=1, length=51):
    max_n_neighbors = min(length, 51)
    #    print('n_neighbors here....... ',max_n_neighbors)
    max_depth = [int(x) for x in range(1, max_n_neighbors)]
    min_samples_split = [int(x) for x in range(2, 11)]

    params = dict(max_depth=max_depth, min_samples_split=min_samples_split)

    grid = GridSearchCV(RandomForestClassifier(n_estimators=100),
                        param_grid=params,
                        cv=low_cv)

    rndmForest_result = train_classifier(grid, training_data, label_data,
                                         full_map)
    return rndmForest_result


def scale_dimension(dimension, factors):
    return round((dimension - factors.get('mean')) / factors.get('stdev'), 5)


def normalize_training_data(training_data):
    data_length = len(training_data)
    scaling_factors = []
    # get the mean and stdev for each dimension
    for x in range(len(training_data[0])):
        mean_for_dimension = round(
            sum([training_data[index][x]
                 for index in range(data_length)]) / data_length, 4)
        dimension_stdev = round(
            (sum([(training_data[index][x] - mean_for_dimension)**2
                  for index in range(data_length)]) / data_length)**0.5, 4)

        scaling_factors.append({
            'mean': mean_for_dimension,
            'stdev': dimension_stdev
        })

    for x in range(len(training_data)):
        training_data[x] = [
            scale_dimension(training_data[x][dimension_index],
                            scaling_factors[dimension_index])
            for dimension_index in range(len(training_data[x]))
        ]
    return training_data


#    weight_mean = round(sum([examples[index][1] for index in range(data_length)])/data_length, 4)
#    weight_stdev = round((sum([(examples[index][1] - weight_mean)**2 for index in range(data_length)])/data_length)**0.5, 4)
#    weight_scaled = [round((examples[index][1] - weight_mean)/weight_stdev, 5) for index in range(data_length)]
def run_test(method,
             training_data,
             label_data,
             low_cv=1,
             length=51,
             range_vals='Sample'):

    if method == 'Sample Data':
        return {'test_data': coords, 'result': colors}

    if range_vals == 'Sample':
        full_map = set_map(sample_range)
    else:
        #        print(range_vals)
        full_map = set_map(range_vals)

    final_result = None

    if method == 'Linear':
        final_result = Lin_Kernel(training_data, label_data, full_map, low_cv)
    if method == 'Polynomial':
        final_result = Poly_Kernel(training_data, label_data, full_map, low_cv)
    if method == 'RBF':
        final_result = RBF_Kernel(training_data, label_data, full_map, low_cv)
    if method == 'Log Regression':
        final_result = Log_Regress(training_data, label_data, full_map, low_cv)
    if method == 'KNN':
        final_result = KNN(training_data, label_data, full_map, low_cv, length)
    if method == 'Tree':
        final_result = Dec_Tree(training_data, label_data, full_map, low_cv,
                                length)
    if method == 'Forest':
        final_result = Rndm_Forest(training_data, label_data, full_map, low_cv,
                                   length)

    return final_result


settings = getKeys(os)


def quote_not_in_collection(collection, quote, search_term):
    results = collection.find_one({search_term: quote.get(search_term)})
    #print( results )
    return not results


def get_next_Value(collection, sequence_name, value):
    sequence = collection.find_one_and_update(
        {'_id': sequence_name}, {'$inc': {
            'sequence_value': value
        }})
    return sequence.get('sequence_value')


def get_date_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_date_obj(date_time_str):
    return datetime.datetime.strptime(date_time_str, '%b %d %Y %I:%M')


def connect_db():
    clientString = settings.get('MONGO_STRING').format(
        settings.get('MONGO_USER'), settings.get('MONGO_USER_PW'),
        'retryWrites=true')
    return MongoClient(clientString)


def db_name():
    return settings.get('DB_NAME')


def connect_to_db(method, args):
    client = None
    return_value = None
    try:
        client = connect_db()
        database = db_name()
        mydb = client[database]
        return_value = method(mydb, args)

    except Exception as err:
        print(err)
        error_collection = mydb[settings.get('ERROR_DB')]
        error_collection.insert_one({
            'error': str(err),
            'date/time': get_date_time()
        })
        return_value = False

    finally:
        if client:
            client.close()
        return return_value


def get_new_quotes(collection, client):
    parser = Parser(**parser_dependencies)
    parser.make_request()
    new_quotes = parser.get_recent_quotes()
    try:
        for quote in new_quotes:
            if quote_not_in_collection(collection, quote, 'quote'):
                quote['_id'] = get_next_Value(collection, 'quote_id', 1)
                collection.insert_one(quote)
    except Exception as err:
        print(err)
    finally:
        if client:
            client.close()


def set_generator(mydb, set_value):

    mycollection = mydb['quotes']
    result = mycollection.update_one({'_id': 'generate_database'},
                                     {'$set': {
                                         'generate': set_value
                                     }})
    return result.matched_count > 0


"""
parsed_line['_id'] = get_next_Value(mycol, 'quote_id')
x = mycol.insert_one(parsed_line)
print(x.inserted_id)
"""
@app.route('/get_quotes/shutdown_generator', methods=["GET"])
def shutdown_generator():
    pw = request.args.get('pw')
    if pw == settings.get('GENERATOR_PW'):
        value = connect_to_db(set_generator, False)
        if value:
            return jsonify({'success': 'generate set to false'})
        else:
            return jsonify({'error': 'something went wrong'})
    else:
        return jsonify({'error': 'invalid command'})


@app.route('/get_quotes/start_generator', methods=["GET"])
def start_generator():
    pw = request.args.get('pw')
    if pw == settings.get('GENERATOR_PW'):
        value = connect_to_db(set_generator, True)
        if value:
            return jsonify({'success': 'generate set to true'})
        else:
            return jsonify({'error': 'something went wrong'})
    else:
        return jsonify({'error': 'invalid command'})


@app.route('/get_quotes/random')
def get_random_quote():
    client = None
    try:
        client = connect_db()
        database = db_name()
        mydb = client[database]
        mycol = mydb['quotes']
        max_index = get_next_Value(mycol, 'quote_id', 0)
        to_return_num = request.args.get('num', '1')
        try:
            to_return_num = int(to_return_num)
        except ValueError:
            return jsonify({'error': 'invalid num argument'})

        generate_database = mycol.find_one({'_id': 'generate_database'})
        if generate_database.get('generate'):

            new_thread = Thread(target=get_new_quotes,
                                args=(mycol, client),
                                daemon=True)
            new_thread.start()

        return_list = []
        for x in range(to_return_num):
            random_index = randint(0, max_index)
            results = mycol.find({'_id': random_index})
            return_list.append(results[0])

        return jsonify(return_list) if len(return_list) > 1 else jsonify(
            return_list[0])
    except Exception as err:
        error_collection = mydb[settings.get('ERROR_DB')]
        error_collection.insert_one({
            'error': str(err),
            'date/time': get_date_time()
        })
        return jsonify({'error': 'Something went wrong'})

    finally:
        if client:
            client.close()


def get_author(mydb, author):
    mycol = mydb['quotes']
    results = mycol.find({'author': author})
    return jsonify(list(results))


@app.route('/get_quotes/find_author', methods=["GET"])
def find_author():
    author = request.args.get('author')
    if not author:
        return jsonify({'error': 'no author provided'})
    result = connect_to_db(get_author, author)
    if result:
        return result
    return jsonify({'error': 'something went wrong'})


def get_source(mydb, source):
    mycol = mydb['quotes']
    results = mycol.find({'source': source})
    return jsonify(list(results))


@app.route('/get_quotes/find_source', methods=["GET"])
def find_source():
    source = request.args.get('source')
    if not source:
        return jsonify({'error': 'no source provided'})
    result = connect_to_db(get_source, source)
    if result:
        return result
    return jsonify({'error': 'something went wrong'})


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route("/svm_visualizer", methods=["POST", "GET"])
def svm_visualizer():
    return render_template("svm_visualizer.html")


@app.route('/get_quotes', methods=["GET", "POST"])
def get_quotes():
    return render_template('quotes_API.html')


@app.route("/get_model/", methods=["POST", "GET"])
def get_model():
    method = request.args.get('runMethod', '')
    data_set = request.args.get('data_set')
    #    print("args ", data_set, " ", method)
    training_data = [[]]
    label_data = []
    low_cv = 5
    length = 51
    vals = 'Sample'

    if data_set == 'Sample':
        #        print("data set sample")
        training_data = coords
        label_data = colors
    else:

        (training_data, label_data, low_cv, vals) = parse_request(data_set)
        length = int(len(label_data) / 2)

    results = run_test(method, training_data, label_data, low_cv, length, vals)
    final_data = {
        "test_data": results.get('test_data').tolist(),
        'result': results.get('result').tolist(),
        'confidence': results.get('confidence'),
        'score': results.get('score'),
        'params': results.get('params')
    }
    #print(final_data)
    return jsonify(final_data)


def parser_new_quotes():
    pass


"""
@app.route('/start_scraper', methods=["GET"])
def start_scraper():
    DAY_TO_SECONDS = 86400
    WAIT_TIME_SECONDS = 60

@app.route('/stop_scraper', methods=["GET"])
def stop_quotes():
    pass    
"""
@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)


def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path, endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['GET'])
def shutdown():
    request_pin = request.args.get('pw')
    if request_pin == settings.get('SHUT_DOWN'):
        shutdown_server()
        return 'Server shutting down...'
    else:
        return 'Invalid request...'


def update_queries(collection, query, new_values):
    # query must be dict( {'search property' : 'search value'})
    # new_values is dict {'propert to change' : 'value to set to'}
    set_values = {'$set': new_values}
    num_changed = collection.update_many(query, set_values)
    return num_changed


if __name__ == "__main__":
    app.debug = True
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)