import numpy as np
import utils
import operator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from dataset import Doc


def eval_err_rate(doc_embeddings, targets):
    random_over_sampler = RandomOverSampler(random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(doc_embeddings, targets, test_size=0.2, random_state=0,
                                                        stratify=targets)
    x_train, y_train = random_over_sampler.fit_resample(x_train, y_train)
    err_rate = sklearn_logistic(x_train, x_test, y_train, y_test)
    return err_rate


def sklearn_logistic(x_train, x_test, y_train, y_test):
    # Train
    logistic = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000)
    logistic.fit(x_train, y_train)

    # Predict & evaluate
    test_predictions = logistic.predict(x_test)
    corrects = sum(np.rint(test_predictions) == [truth for truth in y_test])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return error_rate


def evaluation():
    # load
    all_docs = utils.load('all_docs')
    targets = [doc.target for doc in all_docs]

    doc_emb_dict = utils.load_all_doc_embeddings()
    print('doc embeddings num:', len(doc_emb_dict))

    # evaluate
    result_dict = {}
    count = 1
    for key, value in doc_emb_dict.items():
        print('evaluating no.', count)
        result_dict[key] = eval_err_rate(value, targets)
        count += 1

    # sort
    print('results:')
    result_list = []
    sorted_list = sorted(result_dict.items(), key=operator.itemgetter(1))

    for i, val in enumerate(sorted_list):
        result_list.append((val[0], val[1], i + 1))

    result_list = sorted(result_list, key=operator.itemgetter(0))
    for result in result_list:
        print(result)
