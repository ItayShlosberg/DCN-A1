import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import helpers.dataloader_utils as dataloader_utils
from collections import Counter



class KNNClassifier(object):
    def __init__(self, k):
        self.k = k
        self.x_train = None
        self.y_train = None
        self.n_classes = None

    def train(self, dl_train: DataLoader):
        """
        Trains the KNN model. KNN training is memorizing the training data.
        Or, equivalently, the model parameters are the training data itself.
        :param dl_train: A DataLoader with labeled training sample (should
            return tuples).
        :return: self
        """

        x_train, y_train = dataloader_utils.flatten(dl_train)
        self.x_train = x_train
        self.y_train = y_train
        self.n_classes = len(set(y_train.numpy()))
        return self

    def predict(self, x_test: Tensor):
        """
        Predict the most likely class for each sample in a given tensor.
        :param x_test: Tensor of shape (N,D) where N is the number of samples.
        :return: A tensor of shape (N,) containing the predicted classes.
        """

        # Calculate distances between training and test samples
        dist_matrix = self.calc_distances(x_test)

        # TODO: Implement k-NN class prediction based on distance matrix.
        # For each training sample we'll look for it's k-nearest neighbors.
        # Then we'll predict the label of that sample to be the majority
        # label of it's nearest neighbors.

        n_test = x_test.shape[0]
        y_pred = torch.zeros(n_test, dtype=torch.int64)

        for i in range(n_test):
            # TODO:
            # - Find indices of k-nearest neighbors of test sample i
            # - Set y_pred[i] to the most common class among them

            # ====== YOUR CODE: ======
            # get the labels of the knn.
            dists = dist_matrix[i]
            _, inds = torch.sort(dists, -1, False)
            knn_labels = self.y_train[inds[:self.k]]

            # get the most common label.
            y_pred[i] = int(Counter(knn_labels.numpy()).most_common(1)[0][0])
            # ========================

        return y_pred

    def calc_distances(self, x_test: Tensor):
        """
        Calculates the L2 distance between each point in the given test
        samples to each point in the training samples.
        :param x_test: Test samples. Should be a tensor of shape (Ntest,D).
        :return: A distance matrix of shape (Ntrain,Ntest) where Ntrain is the
            number of training samples. The entry i, j represents the distance
            between training sample i and test sample j.
        """

        # TODO: Implement L2-distance calculation as efficiently as possible.
        # Notes:
        # - Use only basic pytorch tensor operations, no external code.
        # - No credit will be given for an implementation with two explicit
        #   loops.
        # - Partial credit will be given for an implementation with only one
        #   explicit loop.
        # - Full credit will be given for a fully vectorized implementation
        #   (zero explicit loops). Hint: Open the expression (a-b)^2.

        dists = torch.tensor([])
        # ====== YOUR CODE: ======
        dists = torch.sqrt(((x_test ** 2).sum(1, keepdim=True) +
                            (self.x_train ** 2).sum(1) - 2 * torch.mm(x_test, torch.t(self.x_train))))

        # ========================

        return dists


def accuracy(y: Tensor, y_pred: Tensor):
    """
    Calculate prediction accuracy: the fraction of predictions in that are
    equal to the ground truth.
    :param y: Ground truth tensor of shape (N,)
    :param y_pred: Predictions vector of shape (N,)
    :return: The prediction accuracy as a fraction.
    """
    assert y.shape == y_pred.shape
    assert y.dim() == 1

    # TODO: Calculate prediction accuracy. Don't use an explicit loop.

    accuracy = None
    # ====== YOUR CODE: ======
    accuracy = float((y == y_pred).sum().item())/len(y)
    # ========================

    return accuracy


def find_best_k(ds_train: Dataset, k_choices, num_folds):
    """
    Use cross validation to find the best K for the kNN model.

    :param ds_train: Training dataset.
    :param k_choices: A sequence of possible value of k for the kNN model.
    :param num_folds: Number of folds for cross-validation.
    :return: tuple (best_k, accuracies) where:
        best_k: the value of k with the highest mean accuracy across folds
        accuracies: The accuracies per fold for each k (list of lists).
    """

    accuracies = []

    for i, k in enumerate(k_choices):
        model = KNNClassifier(k)

        # TODO: Train model num_folds times with different train/val data.
        # Don't use any third-party libraries.
        # You can use your train/validation splitter from part 1 (even if
        # that means that it's not really k-fold CV since it will be a
        # different split each iteration), or implement something else.

        # ====== YOUR CODE: ======
        ds_size = len(ds_train)
        fold_size = int(np.ceil(ds_size/num_folds))
        acc_fold = np.zeros(num_folds)
        for fold_idx in range(num_folds):
            # Extract data to train and validation.
            ind_vl = [aa for aa in range(fold_idx*fold_size, (fold_idx+1)*fold_size)]
            ind_tr = [aa for aa in range(fold_idx*fold_size)] + [aa for aa in range((fold_idx+1)*fold_size, ds_size)]
            ds_tr = torch.utils.data.dataset.Subset(ds_train, ind_tr)
            ds_vl = torch.utils.data.dataset.Subset(ds_train, ind_vl)
            dl_tr = torch.utils.data.DataLoader(ds_tr, 1024)
            dl_vl = torch.utils.data.DataLoader(ds_vl, 1024)
            x_vl, y_vl = dataloader_utils.flatten(dl_vl)

            # train model.
            model.train(dl_tr)
            # get validation predictions.
            y_pred = model.predict(x_vl)
            # check accuracy.
            acc_fold[fold_idx] = accuracy(y_vl, y_pred)
        accuracies.append(acc_fold)
        # ========================

    best_k_idx = np.argmax([np.mean(acc) for acc in accuracies])
    best_k = k_choices[best_k_idx]

    return best_k, accuracies
