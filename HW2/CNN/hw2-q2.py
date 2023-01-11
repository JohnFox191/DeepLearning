#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np

import utils

def computeSize(input_size, kernel_size, padding, stride):
    return (input_size[0], (input_size[1]-kernel_size[0]+2*padding)/stride+1, (input_size[2]-kernel_size[1]+2*padding)/stride+1)

class CNN(nn.Module):
    
    def __init__(self, input_size, n_classes, dropout_prob):
        super(CNN, self).__init__()
        # First Convolutional Layer (5x5 Kernel)
        out_channels1 = 8
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=out_channels1, kernel_size=(5,5), stride=1, padding=(5-1)//2)
        self.relu1 = nn.ReLU()
        # Here input size is preserved, given padding=(kernel_size-1)//2

        # First Maxpool Layer (2x2 Kernel)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        print(input_size)
        input_size2 = computeSize(input_size=input_size, kernel_size=(2,2), padding=0, stride=2)

        # Second Convolutional Layer (3x3 Kernel)
        out_channels2 = 16
        self.conv2 = nn.Conv2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=(3,3), stride=1, padding=0)
        self.relu2 = nn.ReLU()
        mp2_input_size = computeSize(input_size=input_size2, kernel_size=(3,3), padding=0, stride=1)

        # Second Maxpool Layer (2x2 Kernel)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        input_size3 = computeSize(input_size=mp2_input_size, kernel_size=(2,2), padding=0, stride=2)

        # First Affine Transpormation Layer
        self.input_features = int(out_channels2 * input_size3[1] * input_size3[2])
        print("input_features size: " + str(self.input_features))
        output_features1 = 600
        self.fc1 = nn.Linear(in_features=self.input_features, out_features=output_features1)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)

        # Second Affine Transformation Layer
        output_features2 = 120
        self.fc2 = nn.Linear(in_features=output_features1, out_features=output_features2)
        self.relu4 = nn.ReLU()

        # Third Affine Transformation Layer Layer
        self.fc3 = nn.Linear(in_features=output_features2, out_features=n_classes)
        # !Only Decomment if using nn.NLLLoss() isntead of nn.CrossEntropyLoss()->uses logSoftmax internally! self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        z1 = self.conv1(x)
        h1 = self.relu1(z1)
        p1 = self.maxpool1(h1)
        
        z2 = self.conv2(p1)
        h2 = self.relu2(z2)
        p2 = self.maxpool2(h2)

        x3 = p2.view(p2.size(0), -1)
        z3 = self.fc1(x3)
        h3 = self.relu3(z3)
        h3_dout = self.dropout(h3)
        
        z4 = self.fc2(h3_dout)
        h4 = self.relu4(z4)
        
        output = self.fc3(h4)
        return output

def train_batch(X, y, model, optimizer, criterion, **kwargs):
    X = torch.Tensor(X).view(-1, 1, 28, 28)
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    return loss.item()

def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    X = torch.Tensor(X).view(-1, 1, 28, 28) if X.shape[0] > 1 else torch.Tensor(X).view(1, 28, 28)
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.pdf' % (name), bbox_inches='tight')


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def plot_feature_maps(model, train_dataset):
    
    model.conv1.register_forward_hook(get_activation('conv1'))
    
    data, _ = train_dataset[4]
    data.unsqueeze_(0)
    output = model(data.view(1, 28, 28))

    plt.imshow(data.reshape(28,-1)) 
    plt.savefig('original_image.pdf')

    k=0
    act = activation['conv1'].squeeze()
    fig,ax = plt.subplots(2,4,figsize=(12, 8))
    
    for i in range(act.size(0)//3):
        for j in range(act.size(0)//2):
            ax[i,j].imshow(act[k].detach().cpu().numpy())
            k+=1  
            plt.savefig('activation_maps.pdf') 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.8)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    data = utils.load_classification_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    # initialize the model
    model = CNN(input_size=(1, 28, 28), n_classes=10, dropout_prob=opt.dropout)
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.CrossEntropyLoss()
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []

    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y)))
    # plot
    config = "{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    plot_feature_maps(model, dataset)

if __name__ == '__main__':
    main()
