import json
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from .models import SvmResult
from .utils import parse_request, get_testing_map
from typing import Optional
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from enum import Enum
import numpy as np
import numpy.typing as npt
from typing import Any
from dataclasses import dataclass

@dataclass
class TrainingResult:
    testing_data: npt.NDArray
    result: npt.NDArray
    confidence: npt.NDArray
    params: dict
    score: float

Float2DArray = np.ndarray[Any, np.dtype[np.float64]]

class ModelMethod(str, Enum):
    LINEAR = "Linear"
    POLYNOMIAL = "Polynomial"
    RBF = "RBF"
    LOG_REGRESSION = "Log Regression"
    KNN = "KNN"
    TREE = "Tree"
    FOREST = "Forest"



def train_svm_model(
        method: ModelMethod, 
        training_data: npt.NDArray, 
        labels: npt.NDArray, 
        test_data: npt.NDArray,
        low_cv: int = 2) -> TrainingResult:
    """
    Train an SVM model based on the input data and return the result.
    This is a placeholder for the actual implementation.
    """
    length = int(len(labels) / 2)

    method_mapping = {
        "Linear": Lin_Kernel,
        "Polynomial": Poly_Kernel,
        "RBF": RBF_Kernel,
        "Log Regression": Log_Regress,
        "KNN": lambda *args: KNN(*args, length),
        "Tree": lambda *args: Dec_Tree(*args, length),
        "Forest": lambda *args: Rndm_Forest(*args, length),
    }

        # Get the function based on the method name
    func = method_mapping.get(method)

    if not func:
        raise ValueError(f"Unsupported method: {method}")
    
    # Call the function with the required arguments
    return func(training_data, labels, test_data, low_cv)


async def update_db_with_results(db_gen: AsyncSession, 
                                 unique_id: str, 
                                 results: TrainingResult, 
                                 method: ModelMethod):
    """
    Update the database with the training results.
    """
    async for db in db_gen:  # Get the database session
        new_record = SvmResult(
            id=unique_id,
            method=method,
            test_data=json.dumps(results.testing_data.tolist()),  # Example data as a string
            result= json.dumps(results.result.tolist()),  # Example result as a string
            confidence=json.dumps(results.confidence),
            score=results.score,
            params=json.dumps(results.params)
        )
        db.add(new_record)  # Add the new record to the session
        await db.commit()  # Commit the transaction
        await db.refresh(new_record)  # Refresh the instance with updated data from the DB
        print(f"New record inserted with ID: {new_record.id}")


def train_svm_from_data_then_update_db(
    unique_id: str,
    training_data: List[List[float]],
    testing_data: Optional[List[List[float]]],
    labels: List[str],
    method: ModelMethod,
    db_session_factory: callable,
):
    (data_set, labels, low_cv, rang_vals) = parse_request(training_data, labels)
    test_data = get_testing_map(testing_data, rang_vals)
    
    """
    Train the SVM model and update the database after completion.
    """
    # Train the model
    results = train_svm_model(method, data_set, labels, test_data, low_cv)

    # Get a new database session
    db = db_session_factory()

    # Update the database asynchronously
    import asyncio
    asyncio.run(update_db_with_results(db, unique_id, results, method))

def train_classifier(grid,
                        X_train,
                        y_train,
                        full_map,
                        best=True,
                        normalize=False) -> TrainingResult:

    grid.fit(X_train, y_train)

    if best:
        best_score = grid.best_score_
        best_params = grid.best_params_

    test_out = None
    if normalize:
        test_out = grid.predict(preprocessing.scale(full_map))
    else:
        test_out = grid.predict(full_map)
    proba = grid.predict_proba(full_map)
    probability = [max(x.tolist()) for x in proba]

    return TrainingResult(
        testing_data=full_map,
        result=test_out,
        confidence=probability,
        params=best_params,
        score=best_score
    )

def Lin_Kernel(
        training_data: npt.NDArray, 
        label_data: npt.NDArray,
        full_map: npt.NDArray, 
        low_cv=2):

    C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]

    params = dict(C=C_vals)

    sv = SVC(kernel='linear', probability=True)
    grid = GridSearchCV(sv, params, cv=low_cv)

    linear_result = train_classifier(grid, training_data, label_data,
                                            full_map)
    return linear_result

def Poly_Kernel(training_data: npt.NDArray, label_data: npt.NDArray, full_map: npt.NDArray, low_cv=2):

    C_vals = [0.1, 1, 3]
    degree = [4, 5, 6]
    gamma = [0.1, 0.5]

    params = dict(C=C_vals, degree=degree, gamma=gamma)

    grid = GridSearchCV(SVC(kernel='poly', probability=True),
                                param_grid=params,
                                cv=2)
    #    grid = SVC(kernel='poly', gamma=2, C=1, degree=4)
    poly_result = train_classifier(grid, training_data, label_data,
                                        full_map)
    return poly_result

def RBF_Kernel(training_data: npt.NDArray, label_data: npt.NDArray, full_map: npt.NDArray, low_cv=2):
    #    k_folds = 5
    C_vals = [0.1, 0.5, 1, 5, 10, 50, 100]
    gamma = [0.1, 0.5, 1, 3, 6, 10]
    #cv = StratifiedShuffleSplit(n_splits = k_folds, test_size = 0.40, random_state = 0)

    params = dict(C=C_vals, gamma=gamma)
    #    kernel = 1.0 * RBF(1.0)
    #    gpc = GaussianProcessClassifier(kernel=kernel, random_state=0)
    gpc = SVC(probability=True)
    grid = GridSearchCV(gpc, param_grid=params, cv=low_cv)

    rbf_result = train_classifier(grid, training_data, label_data,
                                        full_map)
    return rbf_result

def Log_Regress(training_data: npt.NDArray, label_data: npt.NDArray, full_map: npt.NDArray, low_cv: int = 1):
    C = [0.1, 0.5, 1, 5, 10, 50, 100]

    params = dict(C=C)
    # any method that uses a gradient descent should normalize data set
    training_data = preprocessing.scale(training_data)
    grid = GridSearchCV(LogisticRegression(solver='lbfgs'),
                                param_grid=params,
                                cv=low_cv)
    log_result = train_classifier(grid, training_data, label_data,
                                        full_map, True, True)

    return log_result

def KNN(training_data: npt.NDArray, label_data: npt.NDArray, full_map: npt.NDArray, low_cv=2, length=51):
    max_n_neighbors = min(length, 51)
    #    print('n_neighbors here....... ',max_n_neighbors)
    n_neighbors = [x for x in range(1, max_n_neighbors)]
    leaf_size = [int(x * 5) for x in range(1, 13)]
    #print(n_neighbors)
    params = dict(n_neighbors=n_neighbors, leaf_size=leaf_size)
    #probability=True
    grid = GridSearchCV(KNeighborsClassifier(),
                                param_grid=params,
                                cv=low_cv)

    KNN_result = train_classifier(grid, training_data, label_data,
                                        full_map)
    return KNN_result

def Dec_Tree(training_data: npt.NDArray,
                label_data: npt.NDArray,
                full_map: npt.NDArray,
                low_cv=2,
                length=51):
    max_n_neighbors = min(length, 51)

    max_depth = [int(x) for x in range(1, max_n_neighbors)]
    min_samples_split = [int(x) for x in range(2, 11)]

    params = dict(max_depth=max_depth, min_samples_split=min_samples_split)

    grid = GridSearchCV(DecisionTreeClassifier(),
                                param_grid=params,
                                cv=low_cv)

    decTree_result = train_classifier(grid, training_data, label_data,
                                            full_map)
    return decTree_result

def Rndm_Forest(training_data,
                label_data,
                full_map,
                low_cv=2,
                length=51):
    max_n_neighbors = min(length, 51)

    max_depth = [int(x) for x in range(1, max_n_neighbors)]
    min_samples_split = [int(x) for x in range(2, 11)]

    params = dict(max_depth=max_depth, min_samples_split=min_samples_split)

    grid = GridSearchCV(RandomForestClassifier(n_estimators=100),
                                param_grid=params,
                                cv=low_cv)

    rndmForest_result = train_classifier(grid, training_data,
                                                label_data, full_map)
    return rndmForest_result    