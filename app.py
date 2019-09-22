import os
import os.path
from flask import Flask, request, render_template, url_for, jsonify
from flask_cors import CORS
from flask_jsglue import JSGlue
from tempfile import mkdtemp

## mongo helper dependencies + Parser from helpers
from threading import Thread, Lock
from pymongo import MongoClient
import datetime
from random import randint

## start parser dependencies ################
from requests import get
from re import sub
from bs4 import BeautifulSoup
from random import shuffle
#### start svm dependencies ############
import json
import numpy as np
### end parser dependencies ################
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
import hashlib

from helpers import Parser, train_svm_from_data_then_update_db
from svm import SVM_Helper
from mongo_helper import MongoHelper
from config import getKeys

settings = getKeys(os)

parser_dependencies = {
    "get": get,
    "np": np,
    "BeautifulSoup": BeautifulSoup,
    "json": json,
    "shuffle": shuffle,
    "sub": sub
}
wiki_quote_scraper = Parser(**parser_dependencies)

svm_dependencies = {
    'np': np,
    'json': json,
    "preprocessing": preprocessing,
    "GridSearchCV": GridSearchCV,
    "SVC": SVC
}
svm_dependencies['LogisticRegression'] = LogisticRegression
svm_dependencies['KNeighborsClassifier'] = KNeighborsClassifier
svm_dependencies["DecisionTreeClassifier"] = DecisionTreeClassifier
svm_dependencies["RandomForestClassifier"] = RandomForestClassifier

svm_helper = SVM_Helper(**svm_dependencies)
svm_helper.set_map(svm_helper.sample_range)

mongo_helper_depenecies = {'datetime': datetime, 'MongoClient': MongoClient}
mongo_helper_depenecies['settings'] = settings
mongo_helper_depenecies['Thread'] = Thread
mongo_helper_depenecies['randint'] = randint
mongo_helper_depenecies['scraper'] = wiki_quote_scraper

mongo_helper = MongoHelper(**mongo_helper_depenecies)

app = Flask(__name__)
CORS(app)
jsglue = JSGlue(app)

app.jinja_env.add_extension('jinja2.ext.loopcontrols')
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['TEMPLATES_AUTO_RELOAD'] = True

#    weight_mean = round(sum([examples[index][1] for index in range(data_length)])/data_length, 4)
#    weight_stdev = round((sum([(examples[index][1] - weight_mean)**2 for index in range(data_length)])/data_length)**0.5, 4)
#    weight_scaled = [round((examples[index][1] - weight_mean)/weight_stdev, 5) for index in range(data_length)]


@app.route('/get_quotes/shutdown_generator', methods=["GET"])
def shutdown_generator():
    pw = request.args.get('pw')
    if pw == settings.get('GENERATOR_PW'):
        kwargs = {"set_value": False}
        value = mongo_helper.connect_to_db(mongo_helper.set_generator, kwargs)
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
        kwargs = {"set_value": True}
        value = mongo_helper.connect_to_db(mongo_helper.set_generator, kwargs)
        if value:
            return jsonify({'success': 'generate set to true'})
        else:
            return jsonify({'error': 'something went wrong'})
    else:
        return jsonify({'error': 'invalid command'})


@app.route('/get_quotes/random')
def get_random_quote():
    number_of_quotes_to_return = request.args.get('num', '1')
    try:
        number_of_quotes_to_return = int(number_of_quotes_to_return)
    except ValueError:
        return jsonify('invalid num argument')

    kwargs = {"number_of_quotes": number_of_quotes_to_return}
    result = mongo_helper.connect_to_db(mongo_helper.get_random_quotes, kwargs)

    if result.get("success"):
        mongo_helper.generate_new_quotes()
        return jsonify(result.get("value"))
    return jsonify({"error": "Something went wrong."})


@app.route('/get_quotes/find_author', methods=["GET"])
def find_author():
    author = request.args.get('author')
    if not author:
        return jsonify({'error': 'no author provided'})
    kwargs = {"author": author}
    result = mongo_helper.connect_to_db(mongo_helper.get_author, kwargs)
    if result.get("success"):
        return jsonify(result.get("value"))
    return jsonify({'error': 'something went wrong'})


@app.route('/get_quotes/find_source', methods=["GET"])
def find_source():
    source = request.args.get('source')
    if not source:
        return jsonify({'error': 'no source provided'})
    kwargs = {"source": source}
    result = mongo_helper.connect_to_db(mongo_helper.get_source, kwargs)
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


@app.route("/train_model/", methods=["POST", "GET"])
def get_model():
    method = request.form.get('runMethod', '')
    data_set = request.form.get('data_set')
    #    print("args ", data_set, " ", method)
    training_data = [[]]
    label_data = []
    low_cv = 5
    length = 51
    vals = 'Sample'

    if data_set == 'Sample':
        #        print("data set sample")
        training_data = svm_helper.coords
        label_data = svm_helper.colors
    else:

        (training_data, label_data, low_cv,
         vals) = svm_helper.parse_request(data_set)
        length = int(len(label_data) / 2)

    # need to call .tolist() as numpy array is not serializable
    user_request = {
        "method": method,
        "training_data": training_data.tolist(),
        "label_data": label_data.tolist()
    }
    user_request_repr = json.dumps(user_request)
    request_repr_id = hashlib.sha1(user_request_repr.encode()).hexdigest()

    run_test_kwargs = {
        "method": method,
        "training_data": training_data,
        "label_data": label_data,
        "low_cv": low_cv,
        "length": length,
        "range_vals": vals
    }
    ## start thread to handle training the SVM, return ID so can lookup in db when done.
    new_thread = Thread(target=train_svm_from_data_then_update_db,
                        args=(svm_helper, mongo_helper, run_test_kwargs,
                              request_repr_id),
                        daemon=True)
    new_thread.start()

    return jsonify(request_repr_id)


@app.route("/retrieve_model/<string:model_id>", methods=["GET"])
def retrieve_model(model_id):

    kwargs = {"result_id": model_id}
    retrieve_model_if_done = mongo_helper.connect_to_db(
        mongo_helper.return_svm_result_if_done, kwargs)

    if retrieve_model_if_done.get("success"):
        value_to_return = retrieve_model_if_done.get("value")

        if value_to_return:
            value_to_return["status"] = "Finished"
            return jsonify(value_to_return)

        return jsonify({"status": "Processing"})

    return jsonify({"error": "Something went wrong."})


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