from functools import partial
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from readers.reviews_dataset import ReviewsDataset
import pandas as pd


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
    tfidf_df.to_csv('data/TfidfVectorizer Combinations.csv')

def main():
    # load datasets
    train_dataset = load_dataset("data/truecased_reviews_train.jsonl")
    dev_dataset = load_dataset("data/truecased_reviews_dev.jsonl")
    test_dataset = load_dataset("data/truecased_reviews_test.jsonl")

    # create feature vectors by calling compute_features() with all three datasets as parameters
    compute_features(train_dataset, dev_dataset, test_dataset)

if __name__ == "__main__":
    main()
