import numpy as np
from sklearn.utils.extmath import safe_sparse_dot
from utils import compute_average_accuracy

class GD:
    def __init__(self, max_iter, num_features, eta, lam):
        super().__init__()
        self.max_iter = max_iter
        self.eta = eta
        self.lam = lam
        self.num_features = num_features
        self.weights = None
        self.bias = None
        self.train_acc_list, self.dev_acc_list = [], []
        self.train_loss_list, self.dev_loss_list = [], []

     # TODO: complete implementation
    def init_parameters(self):
        # complete implementation
        self.weights = np.array(np.zeros(self.num_features))
        self.bias = 0

    # TODO: complete implementation
    def train(self, train_X, train_y, dev_X, dev_y):
        self.init_parameters()
        # complete implementation

        # run for each iterations
        for Iter in range(self.max_iter):
            print("\nIteration ", Iter+1)
            # compute gradient at current location
            gw = np.zeros(self.num_features)        # initialise gradient of weights at 0
            gb = 0                  # initialise gradient of bias at 0

            for x in range(train_X.shape[0]):
                train_y_hat = safe_sparse_dot(train_X[x, :], self.weights.T) + self.bias          # wx + b

                # update weight gradient and bias partial derivatives
                gw = gw + 2 * (train_y_hat - train_y[x]) * train_X[x, :]
                gb = gb + 2 * (train_y_hat - train_y[x])

                # update weights and bias
            gw = gw + self.lam * self.weights  # add in regularisation term
            self.weights = self.weights - (self.eta * gw)
            self.bias = self.bias - (self.eta * gb)

            train_prediction, train_loss = self.predict(train_X, train_y)
            self.train_acc_list.append(compute_average_accuracy(train_prediction, train_y))
            self.train_loss_list.append(train_loss)

            # compute development data accuracy and loss
            dev_prediction, dev_loss = self.predict(dev_X, dev_y)
            self.dev_acc_list.append(compute_average_accuracy(dev_prediction, dev_y))
            self.dev_loss_list.append(dev_loss)

            print("After {0} iterations, train acc = {1}, dev acc = {2}".format(Iter + 1,
                     compute_average_accuracy(train_prediction, train_y), compute_average_accuracy(dev_prediction, dev_y)))

    # TODO: complete implementation
    def predict(self, X, y=None):
        predicted_labels = []
        pred_avg_loss = 0.0         # initialise loss
        loss = 0
        # complete implementation
        counter = -1
        for x in range(X.shape[0]):
            counter += 1
            test_y_hat = safe_sparse_dot(X[x, :], self.weights.T) + self.bias
            loss = loss + (((test_y_hat + self.bias) - y[counter]) ** 2)        # calculate loss using the loss function
            if test_y_hat > 0:          # predict labels based on wx + b
                prediction = 1
            else:
                prediction = -1

            predicted_labels.append(prediction)
            pred_avg_loss = loss/X.shape[0]             # calculate average predicted los

        return predicted_labels, pred_avg_loss[0]
