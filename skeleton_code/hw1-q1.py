#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import time

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Use non-interactive backend for headless environments

import numpy as np
import utils


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

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
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """

        y_hat = np.argmax(self.W.dot(x_i))
        if y_hat != y_i:                        
            self.W[y_i] += x_i            
            self.W[y_hat] -= x_i

class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001, l2_penalty=0.0, **kwargs):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        raise NotImplementedError # Q1.2 (a,b)


class MLP(object):
    def __init__(self, n_classes, n_features, hidden_size):
        self.W1 = np.random.normal(0.1, 0.1, (hidden_size, n_features))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.normal(0.1, 0.1, (n_classes, hidden_size))
        self.b2 = np.zeros(n_classes)

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes.
        z1 = np.dot(self.W1, X.T) + self.b1[:, None]
        h1 = np.maximum(0, z1)  
        z2 = np.dot(self.W2, h1) + self.b2[:, None]  
        y_hat = np.argmax(z2, axis=0)

        return y_hat

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, Y, learning_rate=0.001, **kwargs):
        """
        Dont forget to return the loss of the epoch.
        """
        total_loss = 0
        for x, y in zip(X, Y):
            # encode class into a one hot vector
            one_hot_y = np.zeros(6)
            one_hot_y[y] = 1

            # ===== FORWARD PASS =====
            z1 = np.dot(self.W1, x) + self.b1  # hidden layer pre-activation (hidden_size,)            
            h1 = np.maximum(0, z1) # ReLU activation (hidden_size,)            
            
            z2 = np.dot(self.W2, h1) + self.b2  # output layer pre-activation (n_classes,)                        
            z2 -= np.max(z2) # subtract max for numerical stability to be able to do exp

            p = np.exp(z2) / sum(np.exp(z2))            
            p = np.clip(p, 1e-15, 1 - 1e-15) # clip probabilities to avoid 0

            # Cross-entropy loss
            loss = -one_hot_y.dot(np.log(p)) 
            total_loss += loss

            # ===== BACKWARD PASS =====
            # gradient of loss wrt z2 (output layer pre-activation)
            grad_z2 = p - one_hot_y  # (n_classes,)

            # gradient wrt W2 and b2 (hidden-to-output weights and biases)
            grad_W2 = np.outer(grad_z2, h1)  # (n_classes, hidden_size)
            grad_b2 = grad_z2  # (n_classes,)

            # gradient wrt hidden layer output (h1)
            grad_h1 = np.dot(self.W2.T, grad_z2)  # (hidden_size,)

            # gradient wrt hidden layer pre-activation (z1)
            grad_z1 = grad_h1 * (z1 > 0)  # ReLU derivative: 1 if z1 > 0, else 0

            # gradient wrt W1 and b1 (input-to-hidden weights and biases)
            grad_W1 = np.outer(grad_z1, x)  # (hidden_size, n_features)
            grad_b1 = grad_z1  # (hidden_size,)

            # ===== PARAMETER UPDATES =====
            self.W2 -= learning_rate * grad_W2
            self.b2 -= learning_rate * grad_b2
            self.W1 -= learning_rate * grad_W1
            self.b1 -= learning_rate * grad_b1    

        return total_loss / X.shape[0]


def plot(epochs, train_accs, val_accs, filename=None):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()

def plot_loss(epochs, loss, filename=None):
    plt.figure()  # Start a new figure
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def plot_w_norm(epochs, w_norms, filename=None):
    plt.figure()  # Start a new figure
    plt.xlabel('Epoch')
    plt.ylabel('W Norm')
    plt.plot(epochs, w_norms, label='train')
    plt.legend()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=100,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    parser.add_argument('-l2_penalty', type=float, default=0.0,)
    parser.add_argument('-data_path', type=str, default='intel_landscapes.v2.npz',)
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_dataset(data_path=opt.data_path, bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    weight_norms = []
    valid_accs = []
    train_accs = []

    start = time.time()

    print('initial train acc: {:.4f} | initial val acc: {:.4f}'.format(
        model.evaluate(train_X, train_y), model.evaluate(dev_X, dev_y)
    ))
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate,
                l2_penalty=opt.l2_penalty,
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        elif opt.model == "logistic_regression":
            weight_norm = np.linalg.norm(model.W)
            print('train acc: {:.4f} | val acc: {:.4f} | W norm: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1], weight_norm,
            ))
            weight_norms.append(weight_norm)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    elapsed_time = time.time() - start
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print('Training took {} minutes and {} seconds'.format(minutes, seconds))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs, filename=f"Q1-{opt.model}-accs.pdf")
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss, filename=f"Q1-{opt.model}-loss.pdf")
    elif opt.model == 'logistic_regression':
        plot_w_norm(epochs, weight_norms, filename=f"Q1-{opt.model}-w_norms.pdf")
    with open(f"Q1-{opt.model}-results.txt", "w") as f:
        f.write(f"Final test acc: {model.evaluate(test_X, test_y)}\n")
        f.write(f"Training time: {minutes} minutes and {seconds} seconds\n")


if __name__ == '__main__':
    # a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    # a = np.array([1, 2, 3, 4])
    # print(a.shape)
    # print(np.shape(a[:, None]))
    # print(a[:, None])
    main()
