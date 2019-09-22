class MongoHelper:
    def __init__(self, settings, datetime, MongoClient, Thread, randint,
                 scraper):
        self.env = settings
        self.datetime = datetime
        self.MongoClient = MongoClient
        self.Thread = Thread
        self.randint = randint
        self.scraper = scraper

    def quote_not_in_collection(self, collection, quote, search_term):
        results = collection.find_one({search_term: quote.get(search_term)})
        return not results

    def get_next_Value(self, collection, sequence_name, value):
        sequence = collection.find_one_and_update(
            {'_id': sequence_name}, {'$inc': {
                'sequence_value': value
            }})
        return sequence.get('sequence_value')

    def get_date_time(self):
        return self.datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_date_obj(self, date_time_str):
        return self.datetime.datetime.strptime(date_time_str, '%b %d %Y %I:%M')

    def connect_db(self):
        clientString = self.env.get('MONGO_STRING').format(
            self.env.get('MONGO_USER'), self.env.get('MONGO_USER_PW'),
            'retryWrites=true')
        return self.MongoClient(clientString)

    def db_name(self):
        return self.env.get('DB_NAME')

    def connect_to_db(self, method, kwargs=None):
        client = None
        return_value = {'success': False, "value": False}
        try:
            client = self.connect_db()
            database = self.db_name()
            mydb = client[database]
            if kwargs == None:
                return_value['value'] = method(mydb)
            else:
                return_value['value'] = method(mydb, **kwargs)
            return_value['success'] = True
            if client:
                client.close()
            return return_value

        except Exception as err:
            print(err)
            error_collection = mydb[self.env.get('ERROR_DB')]
            error_collection.insert_one({
                'error': str(err),
                'date/time': self.get_date_time()
            })
            if client:
                client.close()
            return return_value

    def get_new_quotes(self, mydb):
        collection = mydb['quotes']
        self.scraper.make_request()
        new_quotes = self.scraper.get_recent_quotes()
        for quote in new_quotes:
            if self.quote_not_in_collection(collection, quote, 'quote'):
                quote['_id'] = self.get_next_Value(collection, 'quote_id', 1)
                collection.insert_one(quote)

    def set_generator(self, mydb, set_value):

        mycollection = mydb['quotes']
        result = mycollection.update_one({'_id': 'generate_database'},
                                         {'$set': {
                                             'generate': set_value
                                         }})
        return result.matched_count > 0

    def get_author(self, mydb, author):
        mycol = mydb['quotes']
        results = mycol.find({'author': author})
        return list(results)

    def get_source(self, mydb, source):
        mycol = mydb['quotes']
        results = mycol.find({'source': source})
        return list(results)

    def get_random_quotes(self, mydb, number_of_quotes):
        collection = mydb['quotes']
        max_index = self.get_next_Value(collection, "quote_id", 0)
        print(max_index)
        return_list = []
        for x in range(number_of_quotes):
            random_index = self.randint(0, max_index)
            find_quote = collection.find_one({'_id': random_index})
            return_list.append(find_quote)
        if len(return_list) == 1:
            return return_list[0]
        return return_list

    def generate_new_quotes(self):
        new_thread = self.Thread(target=self.check_generate,
                                 args=[],
                                 daemon=True)
        new_thread.start()

    def check_generate(self):
        check_if_generate_result = self.connect_to_db(self.check_if_generate)
        if not check_if_generate_result.get("success"):
            return
        if not check_if_generate_result.get("value"):
            return
        self.connect_to_db(self.get_new_quotes)

    def check_if_generate(self, mydb):
        collection = mydb['quotes']
        generate_database = collection.find_one({'_id': 'generate_database'})
        if generate_database.get("generate"):
            return True
        return False

    def inc_svm_sequence_id(self, mydb, value):
        collection = mydb[self.env.get("SVM_DB")]
        sequence = collection.find_one_and_update({'_id': "id_sequence"},
                                                  {'$inc': {
                                                      'value': value
                                                  }})
        return sequence.get('value')

    def insert_trained_data_to_db(self, mydb, svm_result):
        collection = mydb[self.env.get("SVM_DB")]
        collection.insert_one(svm_result)

    def return_svm_result_if_done(self, mydb, result_id):
        collection = mydb[self.env.get("SVM_DB")]
        find_result = collection.find_one({"_id": result_id}, {"_id": 0})
        # print(find_result)
        if find_result:
            return find_result
        return False

    def id_exists_in_collection(self, item_id, collection_name):
        kwargs = {"item_id": item_id, "collection_name": collection_name}
        id_in_db = self.connect_to_db(self.check_for_id, kwargs)
        if id_in_db.get("success"):
            return id_in_db.get("value")
        return True

    def check_for_id(self, mydb, item_id, collection_name):
        collection = mydb[collection_name]
        result = collection.find_one({"_id": item_id})
        if not result:
            return False
        return True