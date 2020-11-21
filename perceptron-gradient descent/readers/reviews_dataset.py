import json
from functools import partial

from sklearn import preprocessing


def return_same_val(x):
    return x


no_tokenizer = partial(return_same_val)


class ReviewsDataset():
    def __init__(self, **kwargs):
        super(ReviewsDataset, self).__init__(**kwargs)
        # This will be initialised by the load method with all the dataset examples
        self.X = None
        self.X_truecased = None
        # This will be initialised by the load method with all the dataset classes
        self.y = None

    def load(self, filename):
        """
        Loads the dataset from the specified filename. The file is in JSONL format which means that
        every line of the file is a JSON object (http://json.org/).

        Each object is represented by the following fields:
            - original_file: file of the original dataset from which this sentence has been extracted
            - sentence: list of tokens for the current sentence
            - sentiment_class: sentiment associated to the current sentence ("positive"/"negative")
        """
        self.X = []
        self.X_truecased = []
        self.y = []

        print("Loading data from filename {}".format(filename))

        with open(filename) as in_file:
            for line in in_file:
                example = json.loads(line.strip())

                self.X.append(example["sentence"])
                self.X_truecased.append(example["truecased_sentence"])
                self.y.append(example["sentiment_class"])
        self.y = preprocessing.label_binarize(self.y, classes=["negative", "positive"], neg_label=-1,
                                              pos_label=1).ravel()

