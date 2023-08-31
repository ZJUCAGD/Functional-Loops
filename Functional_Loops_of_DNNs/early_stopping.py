import numpy as np

from torch import nn, optim
from data import load_dataset
from models import FCN, train_nn_with_val, train_nn_with_functional_topology


def early_stopping(criterion_data: np.array, test_acc: np.array, patience=5, burn_in_rate=0, mode='max'):
    """
    Args:
        criterion_data: early stopping indicator data during training
        test_acc:       test accuracy during training
        patience:       Number of epochs to trigger an early stopping
        burn_in_rate:   Number of epochs to wait before starting to monitor the early stop indicator
        mode:           'max' or 'min'
    """
    counter = 0
    best_score = None
    corresponding_acc = 0
    corresponding_epoch = 0
    number_epoch = len(criterion_data)

    for epoch in range(burn_in_rate, number_epoch):
        score = criterion_data[epoch]
        if best_score is None:
            best_score = score
            corresponding_acc = test_acc[epoch]
            corresponding_epoch = epoch
        elif mode == "max" and score < best_score:
            counter += 1
        elif mode == "min" and score > best_score:
            counter += 1
        else:
            best_score = score
            corresponding_acc = test_acc[epoch]
            corresponding_epoch = epoch
            counter = 0
        if counter >= patience:
            return corresponding_epoch+1, corresponding_acc, 1

    return number_epoch, test_acc[number_epoch-1], 0


def early_stopping_for_all_parameters_models(criterion_data: np.array, test_acc: np.array, feature_mode: str):

    running_times = criterion_data.shape[0]
    num_epochs = criterion_data.shape[1]
    pots = np.zeros((running_times, num_epochs, num_epochs, 3))

    for i in range(running_times):
        single_criterion_data = criterion_data[i, :]
        for b in range(num_epochs):
            for p in range(num_epochs-b):
                pots[i, b, p, :] = early_stopping(
                    single_criterion_data, test_acc[i, :], patience=p+1, burn_in_rate=b, mode=feature_mode)

    return pots


def compare_FP_valloss(criterion_data, target_data):

    num_epochs = criterion_data.shape[1]
    comparison = np.zeros((num_epochs, num_epochs, 2))
    comparison[:, :, :] = np.median(
        criterion_data[:, :, :, 0:2], axis=0)-np.median(target_data[:, :, :, 0:2], axis=0)

    return comparison


def monitoring_val_loss(n_dim, train_iter, val_iter, test_iter, learning_rate, num_epochs):

    is_mlp = True
    nets_train_features = {'train_acc': [], 'val_acc': [], 'test_acc': [
    ], 'train_loss': [], 'val_loss': [], 'test_loss': []}
    for f in nets_train_features.keys():
        nets_train_features[f] = np.zeros((num_nets, num_epochs))

    # Training models on the clear training dataset
    for i in range(num_nets):
        print('This is {:}th Neural Network!'.format(i+1))
        model = FCN(n_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model, single_train_features = train_nn_with_val(model, train_loader=train_iter, val_loader=val_iter, test_loader=test_iter,
                                                         criterion=criterion, optimizer=optimizer, num_epochs=num_epochs, is_mlp=is_mlp)

        # Recording the metrics in the model training
        for f in nets_train_features.keys():
            nets_train_features[f][i, :] = np.array(single_train_features[f])
    return nets_train_features


def monitoring_funtional_loops(n_dim, train_iter, val_iter, test_iter, learning_rate, num_epochs):

    is_mlp = True
    nets_train_features = {'train_acc': [], 'test_acc': [],
                           'train_loss': [], 'test_loss': [], '1L2': []}
    for f in nets_train_features.keys():
        nets_train_features[f] = np.zeros((num_nets, num_epochs))

    # Training models on the clear training dataset
    for i in range(num_nets):
        print('This is {:}th Neural Network!'.format(i+1))
        model = FCN(n_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model, single_train_features = train_nn_with_functional_topology(model, train_loader=train_iter, val_loader=val_iter, test_loader=test_iter,
                                                                         criterion=criterion, optimizer=optimizer, num_epochs=num_epochs, is_mlp=is_mlp)

        # Recording the metrics in the model training
        for f in nets_train_features.keys():
            nets_train_features[f][i, :] = np.array(single_train_features[f])
    return nets_train_features


if __name__ == "__main__":
    batch_size = 32
    num_nets = 20        # Number of models to be trained
    num_epochs = 40      # Number of epochs
    learning_rate = 3e-4
    n_dim = [32*32*3, 800, 300, 800, 10]

    train_iter, val_iter, test_iter = load_dataset(
        batch_size=batch_size, is_val=True, val_size=6000, root='./data', dataset='CIFAR10',
        use_normalize=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    features_with_val = monitoring_val_loss(
        n_dim, train_iter, val_iter, test_iter, learning_rate, num_epochs)

    train_iter, val_iter, test_iter = load_dataset(
        batch_size=batch_size, is_val=False, root='./data', dataset='CIFAR10',
        use_normalize=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    features_with_FP = monitoring_funtional_loops(
        n_dim, train_iter, val_iter, test_iter, learning_rate, num_epochs)

    es_with_val = early_stopping_for_all_parameters_models(
        criterion_data=features_with_val['val_loss'], test_acc=features_with_val['test_acc'], feature_mode='min')
    es_with_FP = early_stopping_for_all_parameters_models(
        criterion_data=features_with_FP['1L2'], test_acc=features_with_FP['test_acc'], feature_mode='max')

    comparison = compare_FP_valloss(es_with_FP, es_with_val)
