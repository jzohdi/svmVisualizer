import os, sys
from pymongo import MongoClient
import datetime
from threading import Thread
from random import randint

from requests import get
from re import sub
from bs4 import BeautifulSoup
from random import shuffle
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mongo_helper import MongoHelper
from helpers import Parser
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

mongo_helper_depenecies = {'datetime': datetime, 'MongoClient': MongoClient}
mongo_helper_depenecies['settings'] = settings
mongo_helper_depenecies['Thread'] = Thread
mongo_helper_depenecies['randint'] = randint
mongo_helper_depenecies['scraper'] = None

mongo_helper = MongoHelper(**mongo_helper_depenecies)


def init_svm_collection(mydb):
    collection = mydb[settings.get("SVM_DB")]
    collection.insert_one({"_id": "id_sequence", "value": 0})


def init_svm():
    mongo_helper.connect_to_db(init_svm_collection)


if __name__ == "__main__":

    methods_to_test = []

    # methods_to_test.append()

    for method in methods_to_test:
        method()