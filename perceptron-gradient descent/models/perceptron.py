import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from utils import compute_average_accuracy

class Perceptron():
    def __init__(self, num_epochs, num_features, averaged, shuf):
        super().__init__()
        self.num_epochs = num_epochs
        self.averaged = averaged
        self.num_features = num_features
        self.shuf = shuf
        self.weights = None
        self.bias = None

    # TODO: complete implementation
    def init_parameters(self):
        # complete implementation
        self.weights = np.zeros(self.num_features)
        self.bias = 0

    # TODO: complete implementation
    def train(self, train_X, train_y, dev_X, dev_y):
        self.init_parameters()
        # create empty vectors
        train_accuracylist, dev_accuracylist = [], []
        train_shuf_accuracylist, dev_shuf_accuracylist = [], []

        # complete implementation
        for epoch in range(self.num_epochs):
            train_prediction, dev_prediction = [], []

            # condition for shuffled dataset
            if self.shuf:
                print("\nEpoch ", epoch + 1, " (shuffled)")
                train_X, train_y = shuffle(train_X, train_y)
            else:
                print("\nEpoch ", epoch + 1, " (original)")

            for x in range(train_X.shape[0]):
                train_activation = safe_sparse_dot(train_X[x, :], self.weights) + self.bias     # wx + b

                if train_activation > 0:
                    prediction = 1
                else:
                    prediction = -1

                train_prediction.append(prediction)     # add predicted values to list

                if train_y[x] * prediction <= 0:        # if actual label != predicted label, product is < 0
                    # update weights and bias
                    self.weights = train_X[x, :].toarray()[0] * train_y[x] + self.weights
                    self.bias += train_y[x]
                else:        # if actual label = predicted label, product is > 0
                    continue

            for x in range(dev_X.shape[0]):         # validate model on development set
                dev_activation = safe_sparse_dot(dev_X[x, :], self.weights) + self.bias

                if dev_activation > 0:
                    prediction = 1
                else:
                    prediction = -1

                dev_prediction.append(prediction)

            # compute average accuracy for train adn development dataset
            train_accuracy = compute_average_accuracy(train_prediction, train_y)
            dev_accuracy = compute_average_accuracy(dev_prediction, dev_y)

            # add to respective lists
            if self.shuf:
                train_shuf_accuracylist.append(train_accuracy)
                dev_shuf_accuracylist.append(dev_accuracy)
            else:
                train_accuracylist.append(train_accuracy)
                dev_accuracylist.append(dev_accuracy)

            # display results
            print("Bias for Epoch {} = {}".format(epoch+1, self.bias))
            print("Training accuracy of Epoch {} = {:.5f}".format(epoch+1, train_accuracy))
            print("Validation accuracy of Epoch {} = {:.5f}".format(epoch+1, dev_accuracy))

        if self.shuf:
            plt.plot(range(self.num_epochs), train_shuf_accuracylist, label="Train (shuffled)")
            plt.plot(range(self.num_epochs), dev_shuf_accuracylist, label="Dev (shuffled)")
            print("\nHighest validation accuracy (shuffled) is {:1.5f} at Epoch {}".format(max(dev_shuf_accuracylist), np.argmax(dev_shuf_accuracylist) + 1))
        else:
            plt.plot(range(self.num_epochs), train_accuracylist, label="Train (original)")
            plt.plot(range(self.num_epochs), dev_accuracylist, label="Dev (original)")
            print("\nHighest validation accuracy (original) is {:1.5f} at Epoch {}".format(max(dev_accuracylist), np.argmax(dev_accuracylist)+1))

    # TODO: complete implementation
    def predict(self, test_X):
        test_prediction = []
        # complete implementation
        # test the model
        for x in range(test_X.shape[0]):
            test_activation = safe_sparse_dot(test_X[x, :], self.weights) + self.bias
            if test_activation > 0:
                prediction = 1
            else:
                prediction = -1

            test_prediction.append(prediction)

        return test_prediction



