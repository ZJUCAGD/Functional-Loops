import numpy as np

from torch import nn, optim
from data import load_dataset, corruption_dataset
from models import FCN, eval_on_dataloader, train_nn
from topology import computing_PD_based_on_cor, get_persistence_features


if __name__ == "__main__":
    batch_size = 32
    num_nets = 20       # Number of models to be trained
    num_epochs = 40      # Number of epochs
    learning_rate = 3e-4
    is_mlp = True
    n_dim = [28*28, 300, 100, 10]

    train_iter, val_iter, test_iter = load_dataset(
        batch_size=batch_size, is_val=False, root='./data', dataset='MNIST', use_normalize=False)

    # Training models on the clear training dataset
    models = []
    for i in range(num_nets):
        print('This is {:}th Neural Network!'.format(i+1))
        model = FCN(n_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model = train_nn(model, train_loader=train_iter, val_loader=val_iter,
                         criterion=criterion, optimizer=optimizer, num_epochs=num_epochs, is_mlp=is_mlp)
        models.append(model)

    n_seq = [0, 50, 100, 200, 300]
    num = np.zeros((len(n_seq), len(models)))
    mean = np.zeros((len(n_seq), len(models)))
    L2 = np.zeros((len(n_seq), len(models)))
    acces = np.zeros((len(n_seq), len(models)))

    # Evaluating the models on the corrupted test set
    for i in range(len(n_seq)):
        if n_seq[i] == 0:
            train_iter, val_iter, corr_test_iter = load_dataset(
                batch_size=batch_size, is_val=False, root='./data', dataset='MNIST', use_normalize=False)
        else:
            corr_test_iter = corruption_dataset(
                32, n=n_seq[i], root='./data', dataset='MNIST', use_normalize=False)

        for j in range(len(models)):
            model = models[j]
            model.eval()
            test_acc, test_loss, all_output_cor = eval_on_dataloader(
                model, corr_test_iter, criterion, is_mlp=is_mlp, is_cor=True)

            # Computing topological properties of functional networks
            dgm = computing_PD_based_on_cor(all_output_cor)
            properties = get_persistence_features([dgm])

            acces[i, j] = test_acc
            num[i, j] = properties['1Number']
            L2[i, j] = properties['1L2']
