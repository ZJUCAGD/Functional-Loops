import time
import torch
import numpy as np

from torch import nn
from torch.autograd import Variable
from topology import computing_PD_based_on_cor, get_persistence_features


class FCN(nn.Module):
    # Definition of a fully connected neural network
    def __init__(self, n_dim):
        # n_dim: list, Dimensions of FCN
        super(FCN, self).__init__()
        self.layers = nn.Sequential()
        for i in range(len(n_dim)-1):
            layer_name = str("layer_%d" % (i))
            self.layers.add_module(
                layer_name, self.FC_layer(n_dim[i], n_dim[i+1]))

    def FC_layer(self, in_dim, out_dim):
        # Definition of a fully connected layer with Leaky RelU
        layers = [nn.Linear(in_dim, out_dim),
                  nn.LeakyReLU()]
        # nn.init.xavier_uniform_(layers[0].weight, gain=nn.init.calculate_gain('leaky_relu'))
        # nn.init.kaiming_normal_(layers[0].weight, mode='fan_in', nonlinearity='leaky_relu')
        return nn.Sequential(*layers)

    def forward_features(self, x):
        # Record the activation values of neurons in each hidden layer
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i == 0:
                self.features = x
            elif not i == len(self.layers)-1:
                self.features = torch.cat((self.features, x), dim=1)
        return x

    def forward(self, x):
        # feedforward calculation
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class TFConvNet(nn.Module):
    # Definition of a convolutional neural network with ReLU for the CIFAR-10 dataset
    # It has the same architecture as https://www.tensorflow.org/tutorials/images/cnn
    def __init__(self, in_channels=3, num_classes=10):
        super(TFConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(inplace=True)
        )

        self.fc_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # feedforward calculation

        x = self.conv_layers(x)
        x = self.fc_classifier(x)

        return x

    def forward_features(self, x):
        # Feedforward calculation
        # Record activation values for each convolution kernel feature map

        conv_features = [2, 5, 7]
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if i in conv_features:
                if i == conv_features[0]:
                    self.features = torch.mean(x, dim=(2, 3))
                else:
                    self.features = torch.cat(
                        (self.features, torch.mean(x, dim=(2, 3))), dim=1)
        x = self.fc_classifier(x)

        return x


def eval_on_dataloader(model, dataloader, criterion, is_mlp=False, is_cor=False):
    # Evaluating networks on datasets

    use_gpu = torch.cuda.is_available()

    layer_output = []

    running_acc = 0.0
    running_loss = 0.0
    num_data = 0
    output_cor = None

    model.eval()
    for data in dataloader:
        img, label = data
        num_data += len(img)
        if is_mlp:
            img = img.view(img.size(0), -1)
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)
        with torch.no_grad():
            if is_cor:
                out = model.forward_features(img)
            else:
                out = model(img)

        # Calculation of the value of the loss
        loss = criterion(out, label)
        running_loss += loss.item()

        # Calculation of accuracy
        _, pred = torch.max(out, 1)
        running_acc += (pred == label).float().sum()

        # Recording of neuronal output values
        if is_cor:
            layer_output = layer_output + model.features.data.cpu().numpy().tolist()

    running_acc = running_acc / num_data * 100
    running_loss = running_loss / num_data

    # Calculate the correlation coefficient matrix
    if is_cor:
        layer_output = np.array(layer_output)
        output_cor = np.corrcoef(layer_output, rowvar=False)

    return running_acc, running_loss, output_cor


def train_nn(model, train_loader, val_loader, criterion, optimizer, num_epochs, is_mlp=True):

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    print('{:<10} {:<10} {:<10} '.format(
        'epoch', 'train_acc', 'train_loss'), end='')
    if val_loader:
        print('{:<10} {:<10} {:<10}'.format('val_acc', 'val_loss', 'times'))
    else:
        print('{:<10}'.format('times'))

    # Start training
    for epoch in range(num_epochs):
        print('{:<10} '.format(epoch + 1), end='')

        since = time.time()

        running_loss = 0.0
        running_acc = 0.0
        num_data = 0

        model.train()
        for data in train_loader:
            img, label = data

            optimizer.zero_grad()

            num_data += len(img)
            if is_mlp:
                img = img.view(img.size(0), -1)
            if use_gpu:
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)

            # forward propagation
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item()
            _, pred = torch.max(out, 1)
            num_correct = (pred == label)
            running_acc += num_correct.float().sum()

            # backward propagation
            loss.backward()
            optimizer.step()

        # Record the output of neurons on the training set and calculate the correlation coefficient matrix
        model.eval()
        train_acc, train_loss, _ = eval_on_dataloader(
            model, train_loader, criterion, is_mlp=is_mlp, is_cor=False)

        print('{:<10.3f} {:<10.6f} '.format(train_acc, train_loss), end='')

        # Calculating network performance on the val set
        if val_loader:
            model.eval()
            val_acc, val_loss, _ = eval_on_dataloader(
                model, val_loader, criterion, is_mlp=is_mlp, is_cor=False)

            print('{:<10.3f} {:<10.6f} '.format(val_acc, val_loss), end='')

        print('{:<.1f}s'.format(time.time() - since))

    return model


def train_nn_with_functional_topology(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, is_mlp=True):

    # Record the accuracy, loss, and functional topology of the DNN at the end of each epoch in the network training process
    features = {'train_acc': None, 'val_acc': None, 'test_acc': None,
                'train_loss': None, 'val_loss': None, 'test_loss': None,
                '1Number': None, '1L2': None, '1Mean': None}

    for index in features.keys():
        features[index] = np.zeros(num_epochs)

    tda_features = ['1Number', '1L2', '1Mean']

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    print('{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format(
        'epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss', 'times'))

    # Start training
    for epoch in range(num_epochs):
        print('{:<10} '.format(epoch + 1), end='')

        since = time.time()

        running_loss = 0.0
        running_acc = 0.0
        num_data = 0

        for data in train_loader:
            model.train()
            img, label = data

            optimizer.zero_grad()

            num_data += len(img)
            if is_mlp:
                img = img.view(img.size(0), -1)
            if use_gpu:
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)

            # forward propagation
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item()
            _, pred = torch.max(out, 1)
            num_correct = (pred == label)
            running_acc += num_correct.float().sum()

            # backward propagation
            loss.backward()
            optimizer.step()

        # Record the output of neurons on the training set and calculate the correlation coefficient matrix
        model.eval()
        train_acc, train_loss, all_output_cor = eval_on_dataloader(
            model, train_loader, criterion, is_mlp=is_mlp, is_cor=True)

        print('{:<10.3f} {:<10.6f} '.format(train_acc, train_loss), end='')

        features['train_acc'][epoch] = train_acc
        features['train_loss'][epoch] = train_loss

        # Computing topological properties of the functional network
        dgm = computing_PD_based_on_cor(all_output_cor)
        properties = get_persistence_features([dgm])
        for prop in tda_features:
            features[prop][epoch] = properties[prop]

        # Calculating network performance on the val set
        if val_loader:
            model.eval()
            val_acc, val_loss, all_output_cor = eval_on_dataloader(
                model, val_loader, criterion, is_mlp=is_mlp, is_cor=False)

            features['val_acc'][epoch] = val_acc
            features['val_loss'][epoch] = val_loss

        # Calculating network performance on the test set
        model.eval()
        test_acc, test_loss, _ = eval_on_dataloader(
            model, test_loader, criterion, is_mlp=is_mlp, is_cor=False)

        features['test_acc'][epoch] = test_acc
        features['test_loss'][epoch] = test_loss
        print('{:<10.3f} {:<10.6f} '.format(test_acc, test_loss), end='')

        print('{:<.1f}s'.format(time.time() - since))

    return model, features


def train_nn_with_val(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, is_mlp=True):

    # Record the accuracy, loss, and functional topology of the DNN at the end of each epoch in the network training process
    features = {'train_acc': None, 'val_acc': None, 'test_acc': None,
                'train_loss': None, 'val_loss': None, 'test_loss': None}

    for index in features.keys():
        features[index] = np.zeros(num_epochs)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model = model.cuda()

    print('{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}'.format(
        'epoch', 'train_acc', 'train_loss', 'test_acc', 'test_loss', 'times'))

    # Start training
    for epoch in range(num_epochs):
        print('{:<10} '.format(epoch + 1), end='')

        since = time.time()

        running_loss = 0.0
        running_acc = 0.0
        num_data = 0

        for data in train_loader:
            model.train()
            img, label = data

            optimizer.zero_grad()

            num_data += len(img)
            if is_mlp:
                img = img.view(img.size(0), -1)
            if use_gpu:
                img = Variable(img).cuda()
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)

            # forward propagation
            out = model(img)
            loss = criterion(out, label)
            running_loss += loss.item()
            _, pred = torch.max(out, 1)
            num_correct = (pred == label)
            running_acc += num_correct.float().sum()

            # backward propagation
            loss.backward()
            optimizer.step()

        # Record the output of neurons on the training set and calculate the correlation coefficient matrix
        model.eval()
        train_acc, train_loss, _ = eval_on_dataloader(
            model, train_loader, criterion, is_mlp=is_mlp, is_cor=False)

        print('{:<10.3f} {:<10.6f} '.format(train_acc, train_loss), end='')

        features['train_acc'][epoch] = train_acc
        features['train_loss'][epoch] = train_loss

        model.eval()
        val_acc, val_loss, _ = eval_on_dataloader(
            model, val_loader, criterion, is_mlp=is_mlp, is_cor=False)

        features['val_acc'][epoch] = val_acc
        features['val_loss'][epoch] = val_loss

        # Calculating network performance on the test set
        model.eval()
        test_acc, test_loss, _ = eval_on_dataloader(
            model, test_loader, criterion, is_mlp=is_mlp, is_cor=False)

        features['test_acc'][epoch] = test_acc
        features['test_loss'][epoch] = test_loss
        print('{:<10.3f} {:<10.6f} '.format(test_acc, test_loss), end='')

        print('{:<.1f}s'.format(time.time() - since))

    return model, features
