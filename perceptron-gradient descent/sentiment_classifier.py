from functools import partial
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from models.perceptron import Perceptron
from models.gd import GD
from readers.reviews_dataset import ReviewsDataset
from utils import compute_average_accuracy
import pandas as pd
import matplotlib.pyplot as plt
import ast

def return_same_val(x):
    return x

no_tokenizer = partial(return_same_val)


def load_dataset(filename):
    dataset = ReviewsDataset()
    dataset.load(filename)
    return dataset

# Part I - Feature Engineering
# In this part you will need to try out different feature representations
# by using scikit-learn CountVectorizer and TfIdfTransformer.
#
# You should first compute features on the training set (using scikit fit_transform() )
# and then apply the vectorizer on the dev and test set (using scikit transform() ).
#
# Input: the train, dev and test instances of class ReviewDataset
# Output: returns 3 things: train_X_features, dev_X_features and test_X_features. These should be the
# feature vectors for the train, dev and test splits.

def compute_features(train, dev, test):
    '''
    # list down parameters for CountVectorizer for Question 1 (a) (b) (e)
    countparams = {
        'vect__binary': (True, False),
        'vect__min_df': (1, 2),
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
        'vect__stop_words': (None, 'english'),
    }

    # create pipeline for CountVectorizer with MultinomialNB
    countclf = Pipeline([
        ('vect', CountVectorizer(tokenizer=no_tokenizer, lowercase=False)),
        ('mnb', MultinomialNB()),
    ])

    # list down parameters for TfidfVectorizer for Question 1 (c) (d)
    tfidfparams = {
        'tfidf__binary': (True, False),
        'tfidf__min_df': (1, 2),
        'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],
        'tfidf__stop_words': (None, 'english'),
        'tfidf__use_idf': (True, False),
    }

    # create pipeline for CountVectorizer with MultinomialNB
    tfidfclf = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=no_tokenizer, lowercase=False)),
        ('mnb', MultinomialNB()),
    ])

    count_gs_clf = GridSearchCV(countclf, countparams)          # cross validation on CountVectorizer using GridSearchCV
    count_gs_clf.fit(train.X, train.y)          # fit train dataset to the model
    print(count_gs_clf.best_score_)         # choose the best score from cross validation outcome
    print(count_gs_clf.best_params_)            # print best combination of parameters

    tfidf_gs_clf = GridSearchCV(tfidfclf, tfidfparams)          # cross validation on TfidfVectorizer using GridSearchCV
    tfidf_gs_clf.fit(train.X, train.y)          # fit train dataset to the model
    print(tfidf_gs_clf.best_score_)         # choose the best score from cross validation outcome
    print(tfidf_gs_clf.best_params_)            # print best combination of parameters

    count_df = pd.DataFrame(count_gs_clf.cv_results_)[['mean_test_score', 'params']]        # save results in dataframe
    tfidf_df = pd.DataFrame(tfidf_gs_clf.cv_results_)[['mean_test_score', 'params']]

    print(count_df)     # display dataframe
    print(tfidf_df)
    count_df.to_csv('data/CountVectorizer Combinations.csv')        # export to .csv file
    tfidf_df.to_csv('data/TfidfVectorizer Combinations.csv')'''

    # The above code is Grid Search Cross Validation which can be found in the file named GridSearchCV.py
    # Using GridSearchCV and Pipeline, we have generated a .csv file to show all combinations and their accuracies.
    # The .csv file below can be obtained in the /data folder
    # To save time, we will import the generated output directly.
    # Note: To run GridSearchCV in this file, uncomment line 39 to 85

    # read file from .csv file containing parameters and their accuracies
    count_df = pd.read_csv('data/CountVectorizer Combinations.csv')
    tfidf_df = pd.read_csv('data/TfidfVectorizer Combinations.csv')
    # print best accuracy
    print("Maximum Validation Accuracy = ", max(max(count_df["mean_test_score"]), max(tfidf_df["mean_test_score"])))
    best_params_count = ast.literal_eval(count_df.iloc[np.argmax("mean_test_score")]['params'])     # best parameters for CountVectorizer
    best_params_tfidf = ast.literal_eval(tfidf_df.iloc[np.argmax("mean_test_score")]['params'])     # best parameters for TfidfVectorizer

    # decide which vectorizer to use based on highest validation accuracy
    # generate vectorizer using best parameters
    if max(count_df['mean_test_score']) > max(tfidf_df['mean_test_score']):
        vectorizer = CountVectorizer(tokenizer=no_tokenizer, lowercase=False,
                                     binary=best_params_count["vect__binary"],
                                     min_df=best_params_count["vect__min_df"],
                                     ngram_range=best_params_count["vect__ngram_range"],
                                     stop_words=best_params_count["vect__stop_words"])
    else:
        vectorizer = TfidfVectorizer(tokenizer=no_tokenizer, lowercase=False,
                                     binary=best_params_tfidf["tfidf__binary"],
                                     min_df=best_params_tfidf["tfidf__min_df"],
                                     ngram_range=best_params_tfidf["tfidf__ngram_range"],
                                     stop_words=best_params_tfidf["tfidf__stop_words"],
                                     use_idf=best_params_tfidf["tfidf__use_idf"])

    # Step 2. Apply the vectorizer on the training, dev and test set.
    train_X_features = vectorizer.fit_transform(train.X)
    dev_X_features = vectorizer.transform(dev.X)
    test_X_features = vectorizer.transform(test.X)

    # Step 3. return feature vectors for train, dev and test set (NOTE: you should return 3 things!!!)
    return train_X_features, dev_X_features, test_X_features

def main():
    # load datasets
    train_dataset = load_dataset("data/truecased_reviews_train.jsonl")
    dev_dataset = load_dataset("data/truecased_reviews_dev.jsonl")
    test_dataset = load_dataset("data/truecased_reviews_test.jsonl")

    # Part I: Feature Engineering
    # Step 1. create feature vectors by calling compute_features() with all three datasets as parameters
    train_vecs, dev_vecs, test_vecs = compute_features(train_dataset, dev_dataset, test_dataset)

    print("Proportion of +1 label: ", np.mean(train_dataset.y == 1))
    print("Proportion of -1 label: ", np.mean(train_dataset.y == -1))

    # Step 2. train a Naive Bayes Classifier (scikit MultinomialNB() )
    # TODO complete implementation
    mnb = MultinomialNB()
    mnb.fit(train_vecs, train_dataset.y)        # fit model to train set

    # Step 3. Check performance
    prediction = mnb.predict(test_vecs)         # test model
    test_acc = compute_average_accuracy(prediction, test_dataset.y)     # compare actual and predicted labels
    print("Test Accuracy = ", test_acc)

    # Question 1(e)
    # calculate remaining vocabulary size
    vectorizere = CountVectorizer(tokenizer=no_tokenizer, lowercase=False, binary=True, min_df=2)
    train_X_features_e = vectorizere.fit_transform(train_dataset.X)
    print("Remaining vocabulary size = ", train_X_features_e.shape[1])

    # Part II: Perceptron Algorithm
    # TODO: Implement the body of Perceptron.train() and Perceptron.predict()
    # parameters for the perceptron model
    num_epochs = 20
    num_features = train_vecs.shape[1]
    averaged = False    # only MSc students should need to touch this!

    # Step 1. Initialise model with hyperparameters
    perceptron = Perceptron(num_epochs, num_features, averaged, shuf=False)

    # Step 2. Train model
    print("Training model for {} epochs".format(num_epochs))
    perceptron.train(train_vecs, train_dataset.y, dev_vecs, dev_dataset.y)          #train model (original)
    plt.xlabel("Epochs"); plt.ylabel("Accuracy")              # plot graph
    plt.title("Perceptron Train & Dev Accuracy (original)"); plt.legend()
    plt.savefig('Perceptron (original).jpg'); plt.show(block=False)

    # Repeat for shuffled datasets
    perceptron_shuf = Perceptron(num_epochs, num_features, averaged, shuf=True)
    print("Training model for {} epochs".format(num_epochs))
    perceptron_shuf.train(train_vecs, train_dataset.y, dev_vecs, dev_dataset.y)     # train model (shuffled)
    plt.xlabel("Epochs"); plt.ylabel("Accuracy")          # plot graph
    plt.title("Perceptron Train & Dev Accuracy (shuffled)"); plt.legend()
    plt.savefig('Perceptron (shuffled).jpg'); plt.show()

    # Step 3. Compute performance on test set
    test_preds = perceptron.predict(test_vecs)          # predict test set using unshuffled trained model
    test_accuracy = compute_average_accuracy(test_preds, test_dataset.y)
    print("\nTest accuracy is: ", test_accuracy)

    # Part III: Gradient Descent
    # TODO: Implement the body of GD.train() and GD.predict()

    # parameters for the gradient descent algorithm
    max_iter = 20
    num_features = train_vecs.shape[1]
    # eta step (Change default value=0 and choose wisely! Double-check CW instructions)
    # lambda term for regularisation (also choose wisely!)

    # Step 1. Initialise model with hyperparameters
    # three sets of combinations of eta and lambda
    linear_model = GD(max_iter=max_iter, num_features=num_features, eta=0.000015, lam=10)
    linear_model2 = GD(max_iter=max_iter, num_features=num_features, eta=0.000009, lam=10)
    linear_model3 = GD(max_iter=max_iter, num_features=num_features, eta=0.000003, lam=100)

    # Step 2. Train model on a subset of the training set (first 10k examples)
    # train model with first set
    print("\nTraining model for {} max_iter".format(max_iter))
    linear_model.train(train_vecs[:10000], train_dataset.y[:10000], dev_vecs, dev_dataset.y)
    plt.plot(range(max_iter), linear_model.train_acc_list, label="Train, eta=0.000015, lam=10")
    plt.plot(range(max_iter), linear_model.dev_acc_list, label="Dev, eta=0.000015, lam=10")

    # train model with second set
    linear_model2.train(train_vecs[:10000], train_dataset.y[:10000], dev_vecs, dev_dataset.y)
    plt.plot(range(max_iter), linear_model2.train_acc_list, label="Train, eta=0.000009, lam=10")
    plt.plot(range(max_iter), linear_model2.dev_acc_list, label="Dev, eta=0.000009, lam=10")

    # train model with third set
    linear_model3.train(train_vecs[:10000], train_dataset.y[:10000], dev_vecs, dev_dataset.y)
    plt.plot(range(max_iter), linear_model3.train_acc_list, label="Train, eta=0.000003, lam=100")
    plt.plot(range(max_iter), linear_model3.dev_acc_list, label="Dev, eta=0.000003, lam=100")

    plt.xlabel("Iterations"); plt.ylabel("Accuracy")  # plot graph
    plt.title("Gradient Descent Train & Dev Accuracy")
    plt.legend(); plt.savefig("GD Accuracy Curve.png")      # plot graph
    plt.show()

    # plot loss curves
    plt.plot(range(max_iter), linear_model.train_loss_list, label="Train, eta=0.000015, lam=10")
    plt.plot(range(max_iter), linear_model.dev_loss_list, label="Dev, eta=0.000015, lam=10")
    plt.plot(range(max_iter), linear_model2.train_loss_list, label="Train, eta=0.000009, lam=10")
    plt.plot(range(max_iter), linear_model2.dev_loss_list, label="Dev, eta=0.000009, lam=10")
    plt.plot(range(max_iter), linear_model3.train_loss_list, label="Train, eta=0.000003, lam=100")
    plt.plot(range(max_iter), linear_model3.dev_loss_list, label="Dev, eta=0.000003, lam=100")
    plt.xlabel("Iterations"); plt.ylabel("Loss")  # plot graph
    plt.title("Gradient Descent Loss Curve")
    plt.legend(); plt.savefig("GD Loss Curve.png")
    plt.show()

    # Step 4. Compute performance on test set
    test_preds, pred_avg_loss = linear_model.predict(test_vecs, test_dataset.y)        # use the model with the best eta and lambda
    test_acc = compute_average_accuracy(test_preds, test_dataset.y)
    print("Test accuracy = ", test_acc)
    print("Predicted Average Loss = ", pred_avg_loss)


if __name__ == "__main__":
    main()
