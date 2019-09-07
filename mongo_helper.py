class MongoHelper:
    def __init__(self, settings, datetime, MongoClient):
        self.env = settings
        self.datetime = datetime
        self.MongoClient = MongoClient

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

    def connect_to_db(self, method, kwargs):
        client = None
        return_value = {'success': False, "value": False}
        try:
            client = self.connect_db()
            database = self.db_name()
            mydb = client[database]
            return_value['success'] = True
            return_value['value'] = method(mydb, **kwargs)

        except Exception as err:
            print(err)
            error_collection = mydb[self.env.get('ERROR_DB')]
            error_collection.insert_one({
                'error': str(err),
                'date/time': self.get_date_time()
            })
        finally:
            if client:
                client.close()
            return return_value

    def get_new_quotes(self, scraper, collection, client):
        scraper.make_request()
        new_quotes = scraper.get_recent_quotes()
        try:
            for quote in new_quotes:
                if self.quote_not_in_collection(collection, quote, 'quote'):
                    quote['_id'] = self.get_next_Value(collection, 'quote_id',
                                                       1)
                    collection.insert_one(quote)
        except Exception as err:
            print("error while getting new quotes...")
            print(err)
        finally:
            if client:
                client.close()

    def set_generator(self, mydb, set_value):

        mycollection = mydb['quotes']
        result = mycollection.update_one({'_id': 'generate_database'},
                                         {'$set': {
                                             'generate': set_value
                                         }})
        return result.matched_count > 0