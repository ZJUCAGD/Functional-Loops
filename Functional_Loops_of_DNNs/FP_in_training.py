import numpy as np

from torch import nn, optim
from data import load_dataset
from models import FCN, train_nn_with_functional_topology


if __name__ == "__main__":
    batch_size = 32
    num_nets = 20       # Number of models to be trained
    num_epochs = 40      # Number of epochs
    learning_rate = 3e-4
    is_mlp = True
    n_dim = [32*32*3, 600, 300, 10]

    train_iter, val_iter, test_iter = load_dataset(
        batch_size=batch_size, is_val=False, root='./data', dataset='SVHN',
        use_normalize=True, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    nets_train_features = {'train_acc': [], 'val_acc': [], 'test_acc': [],
                           'train_loss': [], 'val_loss': [], 'test_loss': [],
                           '1Number': [], '1L2': [], '1Mean': []}
    for f in nets_train_features.keys():
        nets_train_features[f] = np.zeros((num_nets, num_epochs))

    for i in range(num_nets):
        print('This is {:}th Neural Network!'.format(i + 1))

        # Model training
        model = FCN(n_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model, single_train_features = train_nn_with_functional_topology(
            model, train_loader=train_iter, val_loader=val_iter, test_loader=test_iter,
            criterion=criterion, optimizer=optimizer, num_epochs=num_epochs, is_mlp=is_mlp)

        # Recording the metrics in the model training
        for f in nets_train_features.keys():
            nets_train_features[f][i, :] = np.array(single_train_features[f])
