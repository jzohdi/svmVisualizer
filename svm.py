class SVM_Helper:
    def __init__(self, np, json, preprocessing, GridSearchCV, SVC,
                 LogisticRegression, KNeighborsClassifier,
                 DecisionTreeClassifier, RandomForestClassifier):

        ########### DEPENDENCIES ###################
        self.preprocessing = preprocessing
        self.np = np
        self.json = json
        self.GridSearchCV = GridSearchCV
        self.SVC = SVC
        self.LogisticRegression = LogisticRegression
        self.KNeighborsClassifier = KNeighborsClassifier
        self.DecisionTreeClassifier = DecisionTreeClassifier
        self.RandomForestClassifier = RandomForestClassifier

        ######### END DEPENDENCIES #################

        self.full_map_X = []
        self.full_map_Y = []
        self.full_map_coordinates = []

        self.sample_range = {'min_x': 0, 'max_x': 4, 'min_y': 0, 'max_y': 4}
        self.data_set = [
            list(map(float,
                     line.rstrip().split(","))) for line in open("input3.csv")
        ]
        self.data_length = len(self.data_set)

        self.A = [self.data_set[index][0] for index in range(self.data_length)]
        self.B = [self.data_set[index][1] for index in range(self.data_length)]
        self.Y_vals = self.np.array(
            [self.data_set[index][2] for index in range(self.data_length)])

        self.colors = self.np.array([
            'red' if self.Y_vals[i] == 1.0 else 'blue'
            for i in range(self.data_length)
        ])
        self.coords = self.np.array([[self.A[i], self.B[i]]
                                     for i in range(self.data_length)])

    def parse_request(self, data_Set):

        raw_data = self.json.loads(data_Set)
        dimensions = len(raw_data[0]) - 1
        #    print(dimensions)

        min_x = float("inf")
        max_x = float("-inf")
        min_y = float("inf")
        max_y = float("-inf")
        min_z = float("inf")
        max_z = float("-inf")

        if dimensions == 2:
            for data in raw_data:
                min_x = min(float(min_x), float(data[0]))
                max_x = max(float(max_x), float(data[0]))
                min_y = min(float(min_y), float(data[1]))
                max_y = max(float(max_y), float(data[1]))

            x_y = [[float(raw_data[i][0]),
                    float(raw_data[i][1])] for i in range(len(raw_data))]

            labels = [raw_data[i][2] for i in range(len(raw_data))]

            unique_labels = set(labels)
            low_count = min([labels.count(label) for label in unique_labels])
            low_cv = min(5, low_count)

            range_vals = {
                'max_x': max_x,
                'min_x': min_x,
                'max_y': max_y,
                'min_y': min_y
            }
            return (self.np.array(x_y), self.np.array(labels), low_cv,
                    range_vals)

        if dimensions == 3:
            for data in raw_data:
                min_x = min(float(min_x), float(data[0]))
                max_x = max(float(max_x), float(data[0]))
                min_y = min(float(min_y), float(data[1]))
                max_y = max(float(max_y), float(data[1]))
                min_z = min(float(min_z), float(data[2]))
                max_z = max(float(max_z), float(data[2]))

            x_y_z = [[
                float(raw_data[i][0]),
                float(raw_data[i][1]),
                float(raw_data[i][2])
            ] for i in range(len(raw_data))]

            labels = [raw_data[i][3] for i in range(len(raw_data))]

            unique_labels = set(labels)
            low_count = min([labels.count(label) for label in unique_labels])
            low_cv = min(5, low_count)

            range_vals = {
                'max_x': max_x,
                'min_x': min_x,
                'max_y': max_y,
                'min_y': min_y,
                'max_z': max_z,
                'min_z': min_z
            }
            return (self.np.array(x_y_z), self.np.array(labels), low_cv,
                    range_vals)

    def train_classifier(self,
                         grid,
                         X_train,
                         y_train,
                         full_map,
                         best=True,
                         normalize=False):

        grid.fit(X_train, y_train)

        if best:
            best_score = grid.best_score_
            best_params = grid.best_params_

        test_out = None
        if normalize:
            test_out = grid.predict(self.preprocessing.scale(full_map))
        else:
            test_out = grid.predict(full_map)
        proba = grid.predict_proba(full_map)
        probability = [max(x.tolist()) for x in proba]
        return {
            'test_data': full_map,
            'result': test_out,
            'confidence': probability,
            'params': best_params,
            'score': best_score
        }

    def Lin_Kernel(self, training_data, label_data, full_map, low_cv=1):

        C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]

        params = dict(C=C_vals)

        sv = self.SVC(kernel='linear', probability=True)
        grid = self.GridSearchCV(sv, params, cv=low_cv)

        linear_result = self.train_classifier(grid, training_data, label_data,
                                              full_map)
        return linear_result

    def Poly_Kernel(self, training_data, label_data, full_map, low_cv=1):

        C_vals = [0.1, 1, 3]
        degree = [4, 5, 6]
        gamma = [0.1, 0.5]

        params = dict(C=C_vals, degree=degree, gamma=gamma)

        grid = self.GridSearchCV(self.SVC(kernel='poly', probability=True),
                                 param_grid=params,
                                 cv=2)
        #    grid = SVC(kernel='poly', gamma=2, C=1, degree=4)
        poly_result = self.train_classifier(grid, training_data, label_data,
                                            full_map)
        return poly_result

    def RBF_Kernel(self, training_data, label_data, full_map, low_cv=1):
        #    k_folds = 5
        C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]
        gamma = [0.1, 0.5, 1, 3, 6, 10]
        #cv = StratifiedShuffleSplit(n_splits = k_folds, test_size = 0.40, random_state = 0)

        params = dict(C=C_vals, gamma=gamma)
        #    kernel = 1.0 * RBF(1.0)
        #    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
        gpc = self.SVC(probability=True)
        grid = self.GridSearchCV(gpc, param_grid=params, cv=low_cv)

        #print(grid.cv_results_)
        rbf_result = self.train_classifier(grid, training_data, label_data,
                                           full_map)
        return rbf_result

    def Log_Regress(self, training_data, label_data, full_map, low_cv=1):
        C = [0.1, 0.5, 1, 5, 10, 50, 100]

        params = dict(C=C)
        # any method that uses a gradient descent should normalize data set
        training_data = self.preprocessing.scale(training_data)
        grid = self.GridSearchCV(self.LogisticRegression(solver='lbfgs'),
                                 param_grid=params,
                                 cv=low_cv)
        log_result = self.train_classifier(grid, training_data, label_data,
                                           full_map, True, True)

        return log_result

    def KNN(self, training_data, label_data, full_map, low_cv=1, length=51):
        max_n_neighbors = min(length, 51)
        #    print('n_neighbors here....... ',max_n_neighbors)
        n_neighbors = [x for x in range(1, max_n_neighbors)]
        leaf_size = [int(x * 5) for x in range(1, 13)]
        #print(n_neighbors)
        params = dict(n_neighbors=n_neighbors, leaf_size=leaf_size)
        #probability=True
        grid = self.GridSearchCV(self.KNeighborsClassifier(),
                                 param_grid=params,
                                 cv=low_cv)

        KNN_result = self.train_classifier(grid, training_data, label_data,
                                           full_map)
        return KNN_result

    def Dec_Tree(self,
                 training_data,
                 label_data,
                 full_map,
                 low_cv=1,
                 length=51):
        max_n_neighbors = min(length, 51)
        #    print('n_neighbors here....... ',max_n_neighbors)
        max_depth = [int(x) for x in range(1, max_n_neighbors)]
        min_samples_split = [int(x) for x in range(2, 11)]

        params = dict(max_depth=max_depth, min_samples_split=min_samples_split)

        grid = self.GridSearchCV(self.DecisionTreeClassifier(),
                                 param_grid=params,
                                 cv=low_cv)

        decTree_result = self.train_classifier(grid, training_data, label_data,
                                               full_map)
        return decTree_result

    def Rndm_Forest(self,
                    training_data,
                    label_data,
                    full_map,
                    low_cv=1,
                    length=51):
        max_n_neighbors = min(length, 51)
        #    print('n_neighbors here....... ',max_n_neighbors)
        max_depth = [int(x) for x in range(1, max_n_neighbors)]
        min_samples_split = [int(x) for x in range(2, 11)]

        params = dict(max_depth=max_depth, min_samples_split=min_samples_split)

        grid = self.GridSearchCV(self.RandomForestClassifier(n_estimators=100),
                                 param_grid=params,
                                 cv=low_cv)

        rndmForest_result = self.train_classifier(grid, training_data,
                                                  label_data, full_map)
        return rndmForest_result

    def run_test(self,
                 method,
                 training_data,
                 label_data,
                 low_cv=1,
                 length=51,
                 range_vals='Sample'):

        if method == 'Sample Data':
            return {'test_data': self.coords, 'result': self.colors}
        if range_vals == 'Sample':
            full_map = self.set_map(self.sample_range)

        else:
            full_map = self.set_map(range_vals)

        if method == 'Linear':
            return self.Lin_Kernel(training_data, label_data, full_map, low_cv)
        if method == 'Polynomial':
            return self.Poly_Kernel(training_data, label_data, full_map,
                                    low_cv)
        if method == 'RBF':
            return self.RBF_Kernel(training_data, label_data, full_map, low_cv)
        if method == 'Log Regression':
            return self.Log_Regress(training_data, label_data, full_map,
                                    low_cv)
        if method == 'KNN':
            return self.KNN(training_data, label_data, full_map, low_cv,
                            length)
        if method == 'Tree':
            return self.Dec_Tree(training_data, label_data, full_map, low_cv,
                                 length)
        if method == 'Forest':
            return self.Rndm_Forest(training_data, label_data, full_map,
                                    low_cv, length)

    def set_map(self, range_vals):
        full_map_coordinates = []

        low_x = min(0, int(range_vals.get('min_x')))
        high_x = int(range_vals.get('max_x'))
        low_y = min(0, int(range_vals.get('min_y')))
        high_y = int(range_vals.get('max_x'))

        if range_vals.get('max_z', None) == None:
            x_coords = self.np.linspace(low_x, high_x, 35).tolist()
            y_coords = self.np.linspace(low_y, high_y, 35).tolist()
            for x_coord in x_coords:
                for y_coord in y_coords:
                    full_map_coordinates.append([x_coord, y_coord])
            return self.np.array(full_map_coordinates)
        else:
            low_z = min(0, int(range_vals.get('min_z')))
            high_z = int(range_vals.get('max_z'))

            x_coords = self.np.linspace(low_x, high_x, 10).tolist()
            y_coords = self.np.linspace(low_y, high_y, 10).tolist()
            z_coords = self.np.linspace(low_z, high_z, 10).tolist()
            for x_coord in x_coords:
                for y_coord in y_coords:
                    for z_coord in z_coords:
                        full_map_coordinates.append(
                            [x_coord, y_coord, z_coord])
            return self.np.array(full_map_coordinates)

    def normalize_training_data(self, training_data):
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
                self.scale_dimension(training_data[x][dimension_index],
                                     scaling_factors[dimension_index])
                for dimension_index in range(len(training_data[x]))
            ]
        return training_data

    def scale_dimension(self, dimension, factors):
        return round((dimension - factors.get('mean')) / factors.get('stdev'),
                     5)
