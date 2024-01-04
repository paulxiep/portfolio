import pickle

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB

from src.paul_sentiment_analysis.prepare_data import prepare_tfidf_data


def evaluate_sklearn_model(model, x_test, y_test, model_name=''):
    print(model.score(x_test.toarray(), y_test))
    print(f1_score(y_test, model.predict(x_test.toarray()), average='weighted'))
    conf_mat = confusion_matrix(y_test, model.predict(x_test.toarray()))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(f'test_{model_name}.png')
    plt.clf()


def naive_bayes_train(x_train, y_train):
    model = MultinomialNB()
    model.fit(x_train.toarray(), y_train)
    with open('naive_bayes.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model


def logistic_regression_train(x_train, y_train):
    model = LogisticRegression(C=2., penalty="l2", solver="liblinear", dual=False, multi_class="ovr")
    model.fit(x_train.toarray(), y_train)
    with open('logistic_regression.pkl', 'wb') as f:
        pickle.dump(model, f)
    return model


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = prepare_tfidf_data(save_vectorizer='sklearn_vectorizer')
    evaluate_sklearn_model(naive_bayes_train(x_train, y_train), x_test, y_test, model_name='naive_bayes')
    evaluate_sklearn_model(logistic_regression_train(x_train, y_train), x_test, y_test,
                           model_name='logistic_regression')
