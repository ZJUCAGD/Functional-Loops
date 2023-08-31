# DNN
# Functional Loops: Monitoring Functional Organization of Deep Neural Networks Using Algebraic Topology
This is an implementation of functional loops of DNNs on Python3.

![](./Functional_Loops_of_DNNs/doc/pipeline.jpg)

## 0. Requirements

giotto-tda=0.6.0, PyTorch=2.0.1 or newer, numpy, scikit-learn, scipy

## 1. Import the dataset
MNIST, Fashion-MNIST, CIFAR-10, and SVHN datasets can be downloaded in `./data` via pytorch.

## 2. Functional Loops On Shifted Samples
we assess the effect of functional organization on network performance by evaluating the functional loops of DNNs on shifted datasets.

`python ./distribution_shift.py` 

## 3. Functional Persistence During Model Training

We also study the topological evolution in functional networks during model training.

`python ./FP_in_training.py` 

## 4. Early Stopping Based on Functional Persistence

we propose an early stopping criterion based on functional persistence and compare it with validation loss and neural persistence.

`python ./early_stopping.py` 