def getKeys(os):
    try:
        dotenv = '.env.ini'

        with open(dotenv, 'r') as file:
            content = file.readlines()

        content = [line.strip().split('=') for line in content if '=' in line]
        env_vars = dict(content)
        if file:
            file.close()
        return env_vars
    except:
        obj = {}
        keys = [
            "SHUT_DOWN", "STOP_QUOTES", "MONGO_USER", "MONGO_USER_PW",
            "ERROR_DB", "DB_NAME", "GENERATOR_PW", "SVM_DB"
        ]
        for key in keys:
            obj[key] = os.environ.get(key)
        obj["MONGO_STRING"] = "mongodb+srv://{}:{}@quotesdata-jkbvr.mongodb.net/test?{}"
        return obj


#    os.environ.update({"SECRET_KEY" : })