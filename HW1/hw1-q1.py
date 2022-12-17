#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt
import math
import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))
        self.learning_rate = kwargs['learning_rate']

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        print("Results:" ,n_correct/n_possible)
        return n_correct / n_possible


class Perceptron(LinearModel):
    
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # predict value using old weights
        y_hat = self.predict(x_i)
        
        # if prediction is wrong update weights
        if y_hat != y_i:
            # increase weights for correct class
            self.W[y_i] = self.learning_rate * np.add(self.W[y_i],x_i)
            # decrease weights for wrong class
            self.W[y_hat] = self.learning_rate * np.subtract(self.W[y_hat],x_i)


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        label_scores = np.dot(self.W, x_i)
        label_scores = label_scores[:, None]
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1
        label_probability = np.exp(label_scores) / np.sum(np.exp(label_scores))
        self.W += self.learning_rate * (y_one_hot - label_probability) * x_i[None, :]

def Relu(x):
    return x * (x > 0)

def DRelu(x):
    return 1. * (x >= 0)


    
class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, n_layers):
        # Initialize an MLP with a single hidden layer.
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_features = n_features
        self.W1 = np.random.normal(loc=0.1, scale=np.sqrt(0.1), size=(hidden_size , n_features)) # [200, 784]
        self.W2 = np.random.normal(loc=0.1, scale=np.sqrt(0.1), size=(n_classes , hidden_size)) # [10, 200]
        self.B1 = np.zeros((hidden_size, 1)) # (200, 1)
        self.B2 = np.zeros((n_classes, 1)) # (10, 1)

    def update_weight(self, x_i, y_i, learning_rate=0.001):

        # Forward propagation
        Z1 = self.W1.dot(x_i[:, None]) + self.B1 # (200, 784) . (784,1) + (200,1) -> (200,1)
        H1 = Relu(Z1) # -> Input da layer 2 -> (200,1)
        Z2 = self.W2.dot(H1) + self.B2 # (10,200) . (200,1) + (10,1) -> (10,1)
        Z2 = np.subtract(Z2, np.max(Z2))
        y_hat_probabities = np.exp(Z2) / np.sum(np.exp(Z2)) # (10,1)

        y_one_hot = np.zeros((np.size(self.W2, 0), 1)) # (10, 1)
        y_one_hot[y_i] = 1
        # COMPUTE LOSS
        loss = -y_one_hot.T.dot(np.log(y_hat_probabities))

        # Backward propagation
        gradZ2 = y_hat_probabities - y_one_hot # (10, 1) - (10, 1) -> (10, 1)
        gradW2 = gradZ2.dot(H1.T) # (10, 1) * (1, 200) -> (10, 200)

        gradH1 = self.W2.T.dot(gradZ2) # (200, 10) * (10, 1) -> (200, 1)
        gradZ1 = np.multiply(gradH1, DRelu(Z1)) # (200, 1) * (200, 1)
        gradW1 = gradZ1.dot(x_i[None, :]) # (200, 1) * (1, 784) -> (200, 784)

        self.W2 -= learning_rate * gradW2 # (10, 200)
        self.W1 -= learning_rate * gradW1 # (200, 784)
        self.B1 -= learning_rate * gradZ1 # (200,1)
        self.B2 -= learning_rate * gradZ2 # (10,1)

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.        
        Z1 = np.dot(self.W1, X.T) + self.B1.dot(np.full((1,X.shape[0]),1.))  # [200,784] . [784,50k] + (200,50k) -> [200,50k]
        H1 = Relu(Z1) # -> Input da layer 2 (x_next) -> [200, 50k]

        Z2 = np.dot(self.W2, H1) + self.B2.dot(np.full((1,X.shape[0]),1.)) # [10, 200] . [200, 50k] + (10,50k) -> [10, 50k]
        Z2 = np.subtract(Z2, np.max(Z2))
        H2 = np.exp(Z2) / np.sum(np.exp(Z2)) # (10, 50k) 
        return H2.argmax(axis=0) # (50k,)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        print("Accuracy: " + str(n_correct / n_possible))
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, learning_rate)
        
        



def plot(epochs, valid_accs, test_accs,train_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.plot(epochs, train_accs, label='train')
    plt.legend()
    plt.show()


loss = []
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats, learning_rate=0.001)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    train_accs = []
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        #FIXME
        train_X = train_X[:train_X.shape[0]//10]
        train_y = train_y[:train_y.shape[0]//10]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs, train_accs)


if __name__ == '__main__':
    main()
