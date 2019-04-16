import numpy as np
import utils
import operator
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def eval_err_rate(doc_embeddings, train_set, test_set):
    x_train = [doc_embeddings[doc.tag] for doc in train_set]
    y_train = [doc.sentiment for doc in train_set]
    x_test = [doc_embeddings[doc.tag] for doc in test_set]
    y_test = [doc.sentiment for doc in test_set]
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


def sklearn_svm(x_train, x_test, y_train, y_test):
    model = SVC(gamma='scale', max_iter=10000)  # rbf kernel
    model.fit(x_train, y_train)

    test_predictions = model.predict(x_test)
    corrects = sum(np.rint(test_predictions) == [truth for truth in y_test])
    errors = len(test_predictions) - corrects
    error_rate = float(errors) / len(test_predictions)
    return error_rate


def main():
    # load
    train_docs = utils.load('train_docs')
    test_docs = utils.load('test_docs')

    doc_emb_dict = utils.load_all_doc_embeddings()
    print('doc embeddings num:', len(doc_emb_dict))

    # evaluate
    result_dict = {}
    count = 1
    for key, value in doc_emb_dict.items():
        print('evaluating no.', count)
        result_dict[key] = eval_err_rate(value, train_docs, test_docs)
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


if __name__ == '__main__':
    main()
